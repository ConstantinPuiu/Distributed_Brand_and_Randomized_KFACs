import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
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

print('torch.__version__ = {}'.format(torch.__version__))

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import get_dataloader
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_kfac_lean_Kfactors_batchsize import KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import l_rate_function

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct

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
""" Partitioning dataset """
def partition_dataset(collation_fct, data_root_path, dataset, batch_size):
    size = dist.get_world_size()
    #bsz = 256 #int(128 / float(size))
    if dataset in ['MNIST', 'cifar10', 'cifar100', 'imagenet']:
        trainset, testset, num_classes = get_dataloader(dataset = dataset, train_batch_size = batch_size,
                                          test_batch_size = batch_size,
                                          collation_fct = collation_fct, root = data_root_path)
    else:
        raise NotImplementedError('dataset = {} is not implemeted'.format(dataset))
        
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=batch_size,
                                         collate_fn = collation_fct,
                                         shuffle=True)
    
    """testset is preprocessed but NOT split over GPUS and currently NOT EVER USED (only blind training is performed).
    TODO: implement testset stuff and have a TEST at the end of epoch and at the end of training!"""
    return train_set, testset, batch_size, num_classes

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
    batch_size = args.batch_size
    
    ### Other arguments added only after starting CFAR10
    n_epochs = args.n_epochs
    TCov_period = args.TCov_period
    TInv_period = args.TInv_period
    # ====================================================
    
    ### FLAG for efficient wor allocaiton  ###
    work_alloc_propto_EVD_cost = args.work_alloc_propto_EVD_cost
    
    #### for selcting net type ##############
    net_type = args.net_type
    #########################################
    
    ##### for data root path and dataset type ###########
    data_root_path = args.data_root_path
    dataset = args.dataset
    
    if dataset == 'imagenet': # for imagenet, if we selected the corrected version of VGG (1hich is only for CIFAR10, ignore the corrected part)
        if '_corrected' in net_type and 'resnet' in net_type:
            net_type = net_type.replace('_corrected', '')
    
    if dataset == 'MNIST':
        # make sure we did not select a net which cna't run with MNIST< namely anything apart form the simple MNIST net
        if net_type != 'Simple_net_for_MNIST':
            print('rank:{}. Because dataset == MNIST we can only use the Simple_net_for_MNIST net, so overwriting given parameter as such'.format(rank))
        net_type = 'Simple_net_for_MNIST'
    else:
        if net_type == 'Simple_net_for_MNIST':
            print('net_type = Simple_net_for_MNIST is only possible when dataset = MNIST. Changing to default net: VGG16_bn_lmxp')
            net_type = 'VGG16_bn_lmxp'
    ##### END: for data root path #######################
    
    ################################  SCHEDULES ######################################################################
    ### for dealing with PERIOD SCHEDULES
    if args.TInv_schedule_flag == 0: # then it's False
        TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
    else:# if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import TInv_schedule
        if 0 in TInv_schedule.keys(): # overwrite TInv_period
            print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(TInv_period, TInv_schedule[0]))
            TInv_period = TInv_schedule[0]
    
    if args.TCov_schedule_flag == 0: # then it's False
        TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
    else: # if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import TCov_schedule
        if 0 in TCov_schedule.keys(): # overwrite TInv_period
            print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(TCov_period, TCov_schedule[0]))
            TCov_period = TCov_schedule[0]
    
    #########################################
            
    ### for dealing with other parameters SCHEDULES ####
    if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
        KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
    else:
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import KFAC_damping_schedule
    KFAC_damping = KFAC_damping_schedule[0]
    ### TO DO: implement the schedules properly: now only sticks at the first entryforever in all 3
    ################################ END SCHEDULES ###################################################################
    ####
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))

    print('GPU-rank {} : Partitioning dataset ...'.format(rank))
    t_partition_dset_1 = time.time()
    train_set, testset, bsz, num_classes = partition_dataset(collation_fct, data_root_path, dataset, batch_size)
    len_train_set = len(train_set)
    t_partition_dset_2 = time.time()
    print('GPU-rank {} : Done partitioning dataset in {:.2f} s! : len(train_set) = {}'.format(rank, t_partition_dset_2 - t_partition_dset_1, len_train_set))
    
    
    ##################### net selection #######################################
    print('GPU-rank {} : Setting up model (neural netowrk)...'.format(rank))
    model = get_net_main_util_fct(net_type, rank, num_classes = num_classes)
    ##################### END: net selection ##################################

    # wrap the model with DDP
    # device_ids tell DDP where the model is
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters = False)
    print('GPU-rank {} : Done setting up model (neural netowrk)!'.format(rank))
    #################### The above is defined previously

    print('GPU-rank {} : Initializing optimizer...'.format(rank))
    optimizer =  KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = momentum, stat_decay = stat_decay, 
                                kl_clip = kfac_clip, damping = KFAC_damping, 
                                weight_decay = WD, TCov = TCov_period,
                                TInv = TInv_period,
                                work_alloc_propto_EVD_cost = work_alloc_propto_EVD_cost)#    optim.SGD(model.parameters(),
                                #lr=0.01, momentum=0.5) #Your_Optimizer()
    loss_fn = torch.nn.CrossEntropyLoss() #F.nll_loss #Your_Loss() # nn.CrossEntropyLoss()
    # for test loss use: # nn.CrossEntropyLoss(size_average = False)
    print('GPU-rank {} : Done initializing optimizer. Started training...'.format(rank))
    
    tstart = time.time()
    for epoch in range(0, n_epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        #dataloader.sampler.set_epoch(epoch)
        
        ######### setting parameters according to SCHEDULES ###################
        if epoch in TCov_schedule:
            optimizer.TCov =  TCov_schedule[epoch]
        if epoch in TInv_schedule:
            optimizer.TInv = TInv_schedule[epoch]
        if epoch in KFAC_damping_schedule: 
            optimizer.param_groups[0]['damping'] = KFAC_damping_schedule[epoch]
        ######### END: setting parameters according to SCHEDULES ##############
        
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
            #print('rank = {}, epoch = {} at step = {} ({} steps per epoch) has loss.item() = {}'.format(rank, jdx, optimizer.steps, len_train_set, loss.item()))

            loss.backward()
            if jdx == len_train_set - 1 and epoch == n_epochs - 1:
                tend = time.time()
                print('TIME: {:.3f} s. Rank (GPU number) {} at batch {}, total steps optimizer.steps = {}:'.format(tend-tstart, rank, jdx, optimizer.steps) + ', epoch ' +str(epoch + 1) + ', loss: {}\n'.format(str(loss.item())))
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
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Batch size for 1 GPU (total BS for grad is n_gpu x *this). Total BS for K-factors is just *this! (for lean-ness)')
    
    ### Others added only once moved to CIFAR10
    parser.add_argument('--n_epochs', type=int, default = 10, help = 'Number_of_epochs' )
    parser.add_argument('--TCov_period', type=int, default = 20, help = 'Period of reupdating Kfactors (not inverses)' )
    parser.add_argument('--TInv_period', type=int, default = 100, help = 'Period of reupdating K-factor INVERSE REPREZENTATIONS' )
    
    #### for efficient work allocaiton selection
    parser.add_argument('--work_alloc_propto_EVD_cost', type=bool, default = True, help = 'Set to True if allocation in proportion to EVD cost is desired. Else naive allocation of equal number of modules for each GPU is done!' )
    
    ### for selecting net type
    parser.add_argument('--net_type', type=str, default = 'VGG16_bn_lmxp', help = 'Possible Choices: VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected. Simple_net_for_MNIST is also possible and works only for MNIST: changed to VGG16_bn_lmxp if dataset is other than MNIST and the -for_MNIST net is selected' )
    
    ### for dealing with data path (where the dlded dataset is stored) and dataset itself
    parser.add_argument('--data_root_path', type=str, default = '/data/math-opt-ml/', help = 'fill with path to download data at that root path. Note that you do not need to change this based on the dataset, it will change automatically: each dataset will have its sepparate folder witin the root_data_path directory!' )
    parser.add_argument('--dataset', type=str, default = 'cifar10', help = 'Possible Choices: MNIST, cifar10, imagenet. Case sensitive! Anything else will throw an error. Using imagenet with resnet##_corrected net will force the net to turn to resnet##.' )
    
    ############# SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--TInv_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TInv_schedule (schedule dict for TInv) from solver/schedules/KFAC_schedules.py' ) 
    parser.add_argument('--TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TCov_schedule (schedule dict for TCov) from solver/schedules/KFAC_schedules.py' ) 
    ###for dealing with other optimizer schedules
    parser.add_argument('--KFAC_damping_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the KFAC_damping_schedule (schedule dict for KFAC_damping) from solver/schedules/KFAC_schedules.py . If set to 0, a default schedule is used within the main file. Constant values can be easily achieved by altering the schedule to say {0: 0.1} for instance' ) 
    
    ############# END: SCHEDULE FLAGS #################################################
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args()
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for KFAC lean\n'.format(now_start))
    print('\nImportant args were:\n  --work_alloc_propto_EVD_cost = {} ; \n'.format(args.work_alloc_propto_EVD_cost))
    print('\nScheduling flags were: \n --TInv_schedule_flag = {}, --TCov_schedule_flag = {}, --KFAC_damping_schedule_flag = {}'.format(args.TInv_schedule_flag, args.TCov_schedule_flag, args.KFAC_damping_schedule_flag))
    print('\n !! net_type = {}, dataset = {}'.format(args.net_type, args.dataset))
    
    print('\nDoing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))




