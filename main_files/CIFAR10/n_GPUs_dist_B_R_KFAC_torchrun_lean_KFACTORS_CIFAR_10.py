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

print('torch.__version__'.format(torch.__version__))

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

#from true_kfac_FC_project_adaptive_damping import KFACOptimizer #distributed_kfac_simplest_form
from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import get_dataloader
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_B_R_kfac_lean_Kfactors_batchsize import B_R_KFACOptimizer
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
    #os.environ['NCCL_BLOCKING_WAIT'] = '1'
    #os.environ['NCCL_LL_THRESHOLD'] = '0'
    dist.init_process_group(backend = "nccl", timeout = dateT.timedelta(seconds = 120))#, world_size=world_size)
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
    
    ###others only added after starting CIFAR10
    n_epochs = args.n_epochs
    TCov_period = args.TCov_period
    TInv_period = args.TInv_period
    
    ######### BRAND K-fac (also BRSKFAC) specific parameters
    B_R_period = args.B_R_period
    brand_r_target_excess = args.brand_r_target_excess
    brand_update_multiplier_to_TCov = args.brand_update_multiplier_to_TCov
    # ====================================================
    
    
    ### rsvd adaptive rank ##########
    if args.adaptable_rsvd_rank == 0:
        adaptable_rsvd_rank = False
    else:
        adaptable_rsvd_rank = True
    rsvd_rank_adaptation_TInv_multiplier = args.rsvd_rank_adaptation_TInv_multiplier
    rsvd_target_truncation_rel_err = args.rsvd_target_truncation_rel_err
    maximum_ever_admissible_rsvd_rank = args.maximum_ever_admissible_rsvd_rank    
    rsvd_adaptive_max_history = args.rsvd_adaptive_max_history
    # ====================================================
    
    ### B adaptive rank ##########
    if args.adaptable_B_rank == 0:
        adaptable_B_rank = False
    else:
        adaptable_B_rank = True
    B_rank_adaptation_T_brand_updt_multiplier = args.B_rank_adaptation_T_brand_updt_multiplier
    B_target_truncation_rel_err = args.B_target_truncation_rel_err
    maximum_ever_admissible_B_rank = args.maximum_ever_admissible_B_rank    
    B_adaptive_max_history = args.B_adaptive_max_history
    # ===================================================
    
    #### for selcting net type ##############
    net_type = args.net_type
    #########################################
    
    ### added for efficient work allocation
    if args.work_alloc_propto_RSVD_and_B_cost == 0:
        work_alloc_propto_RSVD_and_B_cost = False
    else:
        work_alloc_propto_RSVD_and_B_cost = True
    # ====================================================
        
    ######## added to control whether we B-truncate before or after inversion ###########
    if args.B_truncate_before_inversion == 0:
        B_truncate_before_inversion = False
    else:
        B_truncate_before_inversion = True
    # ====================================================###############################
    
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
    # instantiate the model(it's your own model) and move it to the right device
    if net_type == 'Conv':
        model = get_network('vgg16_bn_less_maxpool', dropout = True, #depth = 19,
                    num_classes = 10,
                    #growthRate = 12,
                    #compressionRate = 2,
                    widen_factor = 1).to(rank)
    elif net_type == 'FC':
        model = get_network('FC_net_for_CIFAR10', dropout = True, #depth = 19,
                    num_classes = 10).to(rank)
    else:
        raise ValueError('Net of type: net_type = {} Not implemented'.format(net_type) )

    # wrap the model with DDP
    # device_ids tell DDP where the model is
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters = False)
    #################### The above is defined previously

    optimizer =  B_R_KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = momentum, stat_decay = stat_decay, 
                                kl_clip = kfac_clip, damping = KFAC_damping, 
                                weight_decay = WD, TCov = KFAC_matrix_update_frequency,
                                TInv = KFAC_matrix_invert_frequency,
                                rsvd_rank = rsvd_rank,
                                oversampling_parameter = oversampling_parameter,
                                rsvd_niter = rsvd_niter,
                                damping_type = damping_type, #'adaptive',
                                clip_type = clip_type,
                                B_R_period = B_R_period, 
                                brand_r_target_excess = brand_r_target_excess,
                                brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov,
                                #added to dea with truncation before inversion
                                B_truncate_before_inversion = B_truncate_before_inversion,
                                # added to deal with eff work alloc
                                work_alloc_propto_RSVD_and_B_cost = work_alloc_propto_RSVD_and_B_cost,
                                # for dealing with adaptable rsvd rank
                                adaptable_rsvd_rank = adaptable_rsvd_rank,
                                rsvd_target_truncation_rel_err = rsvd_target_truncation_rel_err,
                                maximum_ever_admissible_rsvd_rank = maximum_ever_admissible_rsvd_rank,
                                rsvd_adaptive_max_history = rsvd_adaptive_max_history,
                                rsvd_rank_adaptation_TInv_multiplier = rsvd_rank_adaptation_TInv_multiplier,
                                # for dealing with adaptable B rank
                                adaptable_B_rank = adaptable_B_rank,
                                B_rank_adaptation_T_brand_updt_multiplier = B_rank_adaptation_T_brand_updt_multiplier,
                                B_target_truncation_rel_err = B_target_truncation_rel_err,
                                maximum_ever_admissible_B_rank = maximum_ever_admissible_B_rank,    
                                B_adaptive_max_history = B_adaptive_max_history
                                )#    optim.SGD(model.parameters(),
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
            
            #print('RANK {}: the loss is {}'.format(rank, loss))
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
    parser.add_argument('--damping_type', type=str, default = 'adaptive', help = 'type of damping' )
    parser.add_argument('--clip_type', type=str, default = 'non_standard', help = 'Weight decay' )
    #### TO DO: CODE the RSVD to be adaptive, and adaptivity specific to each K-factor
    #### TO DO cond't: preserve the current functionality and add a rsvd_rank_type switch with "adaptive" vs standard 
    
    ### Others added only once moved to CIFAR10
    parser.add_argument('--n_epochs', type=int, default=10, help='Number_of_epochs' )
    parser.add_argument('--TCov_period', type=int, default=20, help='Period of reupdating Kfactors (not inverses) ' )
    parser.add_argument('--TInv_period', type=int, default=100, help='Period of reupdating K-factor INVERSE REPREZENTAITONS' )
    
    ######### BRAND K-fac (also BRSKFAC) specific parameters
    parser.add_argument('--B_R_period', type=int, default=5, help='The factor by which (for Linear layers) the RSVDperiod is larger (lower freuency for higher brand_period)' )
    parser.add_argument('--brand_r_target_excess', type=int, default=0, help='How many more modes to keep in the B-(.) than in the R-(.) reprezentation' )
    parser.add_argument('--brand_update_multiplier_to_TCov', type=int, default=1, help='The factor by which the B-update frequency is LOWER than the frequency at which we reiceve new K-factor information' )
    # ====================================================
    
    ### added to deal with more efficient work allocaiton
    #
    parser.add_argument('--work_alloc_propto_RSVD_and_B_cost', type=int, default=1, help='Do we want to allocate work in proportion to actual RSVD cost, and actual B-update Cost? set to any nonzero if yes. we use int rather than bool as argparse works badly with bool!' ) 
    
    #### added to allow for B-truncating just before inversion as well
    parser.add_argument('--B_truncate_before_inversion', type=int, default=0, help='Do we want to B-truncate just before inversion (more speed less accuracy) If so set to 1 (or anything other than 0). Standard way to deal with bools wiht buggy argparser that only work correctly wiht numbers!' ) 
    
    #### added to deal with RSVD adaptable rank
    parser.add_argument('--adaptable_rsvd_rank', type=int, default = 0, help='Set to any non-zero integer if we want R- adaptable rank. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--rsvd_target_truncation_rel_err', type=float, default=0.033, help='target truncation error in rsvd: the ran will adapt to be around this error (but rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank)' ) 
    parser.add_argument('--maximum_ever_admissible_rsvd_rank', type=int, default=700, help='Rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank' ) 
    parser.add_argument('--rsvd_rank_adaptation_TInv_multiplier', type = int, default = 5, help = 'After rsvd_rank_adaptation_TInv_multiplier * TInv steps we reconsider ranks')
    parser.add_argument('--rsvd_adaptive_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    
    #### added to deal with B- adaptable rank
    parser.add_argument('--adaptable_B_rank', type=int, default = 0, help='Set to any non-zero integer if we want B- adaptable rank. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--B_target_truncation_rel_err', type=float, default=0.033, help='target truncation error in B-update_truncation: the rank will adapt to be around this error (but B-truncation rank has to be strictly below maximum_ever_admissible_B_rank and above 70. Unlike rsvd it is not above 10. That is because using B with very small truncation rank effectively means we carry no information from before, in which case B is pointless. If you need smaller minimum admissible value than 70, edit the corresponding function in the file adaptive_rank_utils.py)' ) 
    parser.add_argument('--maximum_ever_admissible_B_rank', type=int, default=500, help='B-truncation rank has to be strictly below maximum_ever_admissible_B_rank' ) 
    parser.add_argument('--B_rank_adaptation_T_brand_updt_multiplier', type = int, default = 5, help = 'After B_rank_adaptation_T_brand_updt_multiplier * TCov * brand_update_multiplier_TCov steps we reconsider ranks')
    parser.add_argument('--B_adaptive_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    
    ### for selecting net type
    parser.add_argument('--net_type', type=str, default = 'Conv', help = 'type of net: Conv (gives VGG16_bn less maxpool) or FC (gives an adhoc FC net)' )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args()
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for B-R-KFAC lean with  B_R_period= {}, brand_r_target_excess = {}, brand_update_multiplier_to_TCov = {}\n'.format(now_start, args.B_R_period, args.brand_r_target_excess, args.brand_update_multiplier_to_TCov))
    print('Important args were:\n  --work_alloc_propto_RSVD_and_B_cost = {} ; \n--B_truncate_before_inversion = {}; \n--adaptable_rsvd_rank = {}; \n--rsvd_rank_adaptation_TInv_multiplier = {};\n --adaptable_B_rank = {}; \n --B_rank_adaptation_T_brand_updt_multiplier = {};\n'.format(args.work_alloc_propto_RSVD_and_B_cost, args.B_truncate_before_inversion, args.adaptable_rsvd_rank, args.rsvd_rank_adaptation_TInv_multiplier,args.adaptable_B_rank, args.B_rank_adaptation_T_brand_updt_multiplier))
    #print('type of brand_r_target_excess is {}'.format(type(args.brand_r_target_excess)))
    print('Doing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))




