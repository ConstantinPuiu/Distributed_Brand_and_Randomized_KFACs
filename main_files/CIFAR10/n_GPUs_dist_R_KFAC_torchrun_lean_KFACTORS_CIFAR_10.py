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
import datetime as dateT
from torch.utils.data.dataloader import default_collate

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import get_dataloader
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_R_kfac_lean_Kfactors_batchsize import R_KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import l_rate_function
from Distributed_Brand_and_Randomized_KFACs.main_utils.simple_net_libfile_CIFAR_10 import get_network


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
    size = dist.get_world_size()
    bsz = 256 #int(128 / float(size))
    trainset, testset = get_dataloader(dataset = 'cifar10', train_batch_size = bsz,
                                          test_batch_size = bsz,
                                          collation_fct = collation_fct, root='./data_CIFAR10')
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         collate_fn = collation_fct,
                                         shuffle=True)
    
    """testset is preprocessed but NOT split over GPUS and currently NOT EVER USED (only blind training is performed).
    TODO: implement testset stuff and have a TEST at the end of epoch and at the end of training!"""
    return train_set, testset, bsz
    

def cleanup():
    dist.destroy_process_group()


#from torch.nn.parallel import DistributedDataParallel as DDP
def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", timeout = dateT.timedelta(seconds = 120))#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    ### KFAC HYPERPARAMETERS ##
    kfac_clip = args.kfac_clip; #KFAC_damping = 1e-01 #3e-02; 
    stat_decay = args.stat_decay #0.95
    momentum = args.momentum
    WD = args.WD #0.000007 #1 ########  7e-4 got 9.00 +% acc conistently. 7e-06 worked fine too!
    ### RSVD specific params or stuff that 1st appeared in rsvd(clip and damping type)
    rsvd_rank = args.rsvd_rank
    oversampling_parameter = args.oversampling_parameter
    rsvd_niter = args.rsvd_niter
    damping_type = args.damping_type #'adaptive',
    clip_type = args.clip_type
    rank_adaptation_TInv_multiplier = args.rank_adaptation_TInv_multiplier
    
    ###others only added after starting CIFAR10
    n_epochs = args.n_epochs
    TCov_period = args.TCov_period
    TInv_period = args.TInv_period
    
    ##### work allocation
    if args.work_alloc_propto_RSVD_cost == 0 :
        work_alloc_propto_RSVD_cost = False
    else:
        work_alloc_propto_RSVD_cost = True
    
    if args.work_eff_alloc_with_time_measurement == 0:
        work_eff_alloc_with_time_measurement = False
    else:
        work_eff_alloc_with_time_measurement = True
        
    ### rsvd adaptive rank ##########
    if args.adaptable_rsvd_rank == 0:
        adaptable_rsvd_rank = False
    else:
        adaptable_rsvd_rank = True
    rsvd_target_truncation_rel_err = args.rsvd_target_truncation_rel_err
    maximum_ever_admissible_rsvd_rank = args.maximum_ever_admissible_rsvd_rank    
    rsvd_adaptve_max_history = args.rsvd_adaptve_max_history
    # ====================================================
    # ====================================================
    
    ################### SCHEDULES ###### TO DO: MAKE THE SCHEDULES INPUTABLE FORM COMMAND LINE #####################
    # dict to have schedule! eys are epochs: key map to frequency, stuff only changes at keys and then stays constant.
    KFAC_matrix_update_frequency_dict = {0: TCov_period, 5: TCov_period}#, 10: 5, 20: 5, 22: 5, 50: 5}
    KFAC_matrix_invert_frequency_dict = {0: TInv_period, 5: TInv_period}#, 10: 20, 20: 20, 22: 20, 50: 20}
      
    KFAC_damping_dict = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
    ### TO DO: implement the schedules properly: now only sticks at the first entryforever in all 3
    ################################ END SCHEDULES ###################################################################
    KFAC_matrix_update_frequency = KFAC_matrix_update_frequency_dict[0]
    KFAC_matrix_invert_frequency = KFAC_matrix_invert_frequency_dict[0]
    KFAC_damping = KFAC_damping_dict[0]
    ####
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))

    train_set, testset, bsz = partition_dataset(collation_fct)

    # instantiate the model(it's your own model) and move it to the right device
    model = get_network('vgg16_bn_less_maxpool', dropout = True, #depth = 19,
                    num_classes = 10,
                    #growthRate = 12,
                    #compressionRate = 2,
                    widen_factor = 1).to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously

    optimizer =  R_KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = momentum, stat_decay = stat_decay, 
                                kl_clip = kfac_clip, damping = KFAC_damping, 
                                weight_decay = WD, TCov = KFAC_matrix_update_frequency,
                                TInv = KFAC_matrix_invert_frequency,
                                rsvd_rank = rsvd_rank,
                                oversampling_parameter = oversampling_parameter,
                                rsvd_niter = rsvd_niter,
                                work_alloc_propto_RSVD_cost = work_alloc_propto_RSVD_cost,
                                work_eff_alloc_with_time_measurement = work_eff_alloc_with_time_measurement,
                                damping_type = damping_type, #'adaptive',
                                clip_type = clip_type,
                                adaptable_rsvd_rank = adaptable_rsvd_rank,
                                rsvd_target_truncation_rel_err = rsvd_target_truncation_rel_err,
                                maximum_ever_admissible_rsvd_rank = maximum_ever_admissible_rsvd_rank,
                                rsvd_adaptve_max_history = rsvd_adaptve_max_history,
                                rank_adaptation_TInv_multiplier = rank_adaptation_TInv_multiplier)#    optim.SGD(model.parameters(),
                              #lr=0.01, momentum=0.5) #Your_Optimizer()
    loss_fn = torch.nn.CrossEntropyLoss() #F.nll_loss #Your_Loss() # nn.CrossEntropyLoss()
    # for test loss use: # nn.CrossEntropyLoss(size_average = False)
    
    tstart = time.time()
    for epoch in range(0, n_epochs):
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
            ## debug negative loss only
            #if y not in [0,1,2,3,4,5,6,7,8,9]: # checing for data being corrupted? (it wasn't this it was forgetting the softmax)
            #    print('rank = {} has encountered WEIRD label yi = {}'.format(y))
            print('rank = {} at step ={} has loss.item() = {}'.format(rank,optimizer.steps,loss.item()))

            loss.backward()
            if jdx % 150 == 0 and epoch == n_epochs - 1:
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
    
    ### RSVD specific params or stuff that 1st appeared in rsvd(clip and damping type)
    parser.add_argument('--rsvd_rank', type=int, default = 220, help = 'The target rank of RSVD' )
    parser.add_argument('--oversampling_parameter', type=int, default = 10, help = 'the oversampling parameter of RSVD' )
    parser.add_argument('--rsvd_niter', type=int, default = 3, help = '# of power(like) iterations in getting projection subspace for RSVD' )
    parser.add_argument('--damping_type', type=str, default= 'adaptive', help = 'type of damping' )
    parser.add_argument('--clip_type', type=str, default = 'non_standard', help = 'Weight decay' )
    #### TO DO: CODE the RSVD to be adaptive, and adaptivity specific to each K-factor
    #### TO DO cond't: preserve the current functionality and add a rsvd_rank_type switch with "adaptive" vs standard 
    
    ### Others added only once moved to CIFAR10
    parser.add_argument('--n_epochs', type=int, default=10, help='Number_of_epochs' )
    parser.add_argument('--TCov_period', type=int, default=20, help='Period of reupdating Kfactors (not inverses) ' )
    parser.add_argument('--TInv_period', type=int, default=100, help='Period of reupdating K-factor INVERSE REPREZENTAITONS' )
    
    ### added to deal with more efficient work allocaiton
    #
    parser.add_argument('--work_alloc_propto_RSVD_cost', type=int, default=1, help='Do we want to allocate work in proportion to FORECASTED (based on theoretical complexity) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--work_eff_alloc_with_time_measurement', type=int, default=0, help='Do we want to allocate work in proportion to MEASURED (somewhat noisy) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly. Setting this to 1 (TRUE) has no effect if work_alloc_propto_RSVD_cost == False' ) 
    
    #### added to dal with RSVD adaptable rank
    parser.add_argument('--adaptable_rsvd_rank', type=int, default=1, help='Set to any non-zero integer if we want adaptable rank. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--rsvd_target_truncation_rel_err', type=float, default=0.033, help='target truncation error in rsvd: the ran will adapt to be around this error (but rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank)' ) 
    parser.add_argument('--maximum_ever_admissible_rsvd_rank', type=int, default=700, help='Rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank' ) 
    parser.add_argument('--rank_adaptation_TInv_multiplier', type = int, default = 5, help = 'After rank_adaptation_TInv_multiplier * TInv steps we reconsider ranks')
    parser.add_argument('--rsvd_adaptve_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args()
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for R-KFAC lean\n'.format(now_start))
    print('Important args were:\n  --work_alloc_propto_RSVD_cost = {};\n  --work_eff_alloc_with_time_measurement = {};\n  --adaptable_rsvd_rank = {};\n  --rank_adaptation_TInv_multiplier = {}\n'.format(args.work_alloc_propto_RSVD_cost, args.work_eff_alloc_with_time_measurement, args.adaptable_rsvd_rank, args.rank_adaptation_TInv_multiplier))
    
    print('Doing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))





