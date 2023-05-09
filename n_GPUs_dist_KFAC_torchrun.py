import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from simple_net_libfile import Net
#from simple_net_libfile_2 import Net
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import torchvision.transforms as transforms
import torch
import random as rng
import time
import argparse
from datetime import datetime
from torch.utils.data.dataloader import default_collate

#from true_kfac_FC_project_adaptive_damping import KFACOptimizer #distributed_kfac_simplest_form
from distributed_kfac_simplest_form import KFACOptimizer
from lrfct import l_rate_function


#from torch.utils.data.distributed import DistributedSampler
"""def prepare(rank, world_size, batch_size=128, pin_memory=False, num_workers=0):
    #dataset = Your_Dataset()
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

    return dataloader"""

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        #rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

''' use as'''
""" Partitioning MNIST """
def partition_dataset(collation_fct):
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 256 #int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         collate_fn = collation_fct,
                                         shuffle=True)
    return train_set, bsz

def cleanup():
    dist.destroy_process_group()


#from torch.nn.parallel import DistributedDataParallel as DDP
def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl")#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    ### KFAC HYPERPARAMETERS ##
    kfac_clip = args.kfac_clip; #KFAC_damping = 1e-01 #3e-02; 
    stat_decay = args.stat_decay #0.95
    momentum = args.momentum
    WD = args.WD #0.000007 #1 ########  7e-4 got 9.00 +% acc conistently. 7e-06 worked fine too!
    # ====================================================
    
    ################### SCHEDULES ###### TO DO: MAKE THE SCHEDULES INPUTABLE FORM COMMAND LINE #####################
    # dict to have schedule! eys are epochs: key map to frequency, stuff only changes at keys and then stays constant.
    KFAC_matrix_update_frequency_dict = {0: 5, 5: 5, 10: 5, 20: 5, 22: 5, 50: 5}
    KFAC_matrix_invert_frequency_dict = {0: 20, 5: 20, 10: 20, 20: 20, 22: 20, 50: 20}
      
    KFAC_damping_dict = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
    ### TO DO: implement the schedules properly: now only sticks at the first entryforever in all 3
    ################################ END SCHEDULES ###################################################################
    KFAC_matrix_update_frequency = KFAC_matrix_update_frequency_dict[0]
    KFAC_matrix_invert_frequency = KFAC_matrix_invert_frequency_dict[0]
    KFAC_damping = KFAC_damping_dict[0]
    ####
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))

    train_set, bsz = partition_dataset(collation_fct)

    # instantiate the model(it's your own model) and move it to the right device
    model = Net().to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously

    optimizer =  KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = momentum, stat_decay = stat_decay, 
                            kl_clip = kfac_clip, damping = KFAC_damping, 
                            weight_decay = WD, TCov = KFAC_matrix_update_frequency,
                            TInv = KFAC_matrix_invert_frequency)#    optim.SGD(model.parameters(),
                              #lr=0.01, momentum=0.5) #Your_Optimizer()
    loss_fn = F.nll_loss #Your_Loss()
    
    tstart = time.time()
    for epoch in range(0,25):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        #dataloader.sampler.set_epoch(epoch)
        
        if epoch+1 in KFAC_matrix_update_frequency_dict:
            optimizer.TCov =  KFAC_matrix_update_frequency_dict[epoch+1]
        if epoch+1 in KFAC_matrix_invert_frequency_dict:
            optimizer.TInv = KFAC_matrix_invert_frequency_dict[epoch+1]
        if epoch+1 in KFAC_damping_dict: 
            optimizer.param_groups[0]['damping'] = KFAC_damping_dict[epoch+1]
        
        for jdx, (x,y) in enumerate(train_set):#dataloader):
            #print('\ntype(x) = {}, x = {}, x.get_device() = {}\n'.format(type(x), x, x.get_device()))
            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            #label = x['label']
            if optimizer.steps % optimizer.TCov == 0: #KFAC_matrix_update_frequency == 0:
                optimizer.acc_stats = True
            else:
                optimizer.acc_stats = False
                
            try:
                loss = loss_fn(pred, y)#label)
            except:
                print('\ntype(pred) = {}, pred = {}, pred.get_device() = {}\n'.format(type(pred), pred, pred.get_device()))
                print('\ntype(y) = {}, y = {}, y.get_device() = {}\n'.format(type(y),y, y.get_device()))
                loss = loss_fn(pred, y)

            loss.backward()
            if jdx % 150 == 0 and epoch ==24:
                tend = time.time()
                print('TIME: {}. Rank (GPU number) {} at batch {}:'.format(tend-tstart, rank, jdx) + str(dist.get_rank())+ ', epoch ' +str(epoch + 1) + ', loss: {}\n'.format(str(loss.item())))
                #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
                #    f.write('Rank (GPU number) {} at batch {}:'.format(rank, jdx) + str(dist.get_rank())+ ', epoch ' +str(epoch+1) + ', loss: {}\n'.format(str(loss.item())))
            optimizer.step(epoch_number = epoch + 1, error_savepath = None)
    cleanup()
    print('GPU rank = {} of {} is done correctly!'.format(rank, world_size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    
    parser.add_argument('--kfac_clip', type=int, default=7e-2, help='clip factor for Kfac step' )
    parser.add_argument('--stat_decay', type=int, default=0.95, help='the rho' )
    parser.add_argument('--momentum', type=int, default=0.0, help='momentum' )
    parser.add_argument('--WD', type=int, default=7e-4, help='Weight decay' )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args()
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n'.format(now_start))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))





