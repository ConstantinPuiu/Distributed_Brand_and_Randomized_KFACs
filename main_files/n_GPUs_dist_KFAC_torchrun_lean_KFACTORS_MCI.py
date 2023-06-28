import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import time
from datetime import datetime
from torch.utils.data.dataloader import default_collate

print('torch.__version__ = {}'.format(torch.__version__))

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import partition_dataset, cleanup
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_kfac_lean_Kfactors_batchsize import KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import l_rate_function
from Distributed_Brand_and_Randomized_KFACs.main_utils.arg_parser_utils import parse_args, adjust_args_for_0_1_and_compatibility, adjust_args_for_schedules

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct, train_n_epochs, test


def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl")#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    # adjust args: turn 0-1 variables that should be True / False into True / False AND check and correct (args.net_type, args.dataset) combination
    args = adjust_args_for_0_1_and_compatibility(args, rank, solver_name = 'KFAC')
    ###################################### end adjust 0-1 -===> True / False ######################################################################
    
    ################################  SCHEDULES ######################################################################
    args, TInv_schedule, TCov_schedule, KFAC_damping_schedule, KFAC_damping = adjust_args_for_schedules(args, solver_name = 'KFAC')
    ################################ END SCHEDULES ###################################################################
    ####
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))

    print('GPU-rank {} : Partitioning dataset ...'.format(rank))
    t_partition_dset_1 = time.time()
    
    ############################ Partition data ################################
    train_set, test_set, bsz, num_classes = partition_dataset(collation_fct, args.data_root_path, args.dataset, args.batch_size)
    ####################### END : Partition data ###############################
    
    len_train_set = len(train_set)
    t_partition_dset_2 = time.time()
    print('GPU-rank {} : Done partitioning dataset in {:.2f} s! : len(train_set) = {}'.format(rank, t_partition_dset_2 - t_partition_dset_1, len_train_set))
    
    
    ##################### net selection #######################################
    print('GPU-rank {} : Setting up model (neural netowrk)...'.format(rank))
    model = get_net_main_util_fct(args.net_type, rank, num_classes = num_classes)
    ##################### END: net selection ##################################

    # wrap the model with DDP
    # device_ids tell DDP where the model is
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters = False)
    print('GPU-rank {} : Done setting up model (neural netowrk)!'.format(rank))
    #################### The above is defined previously

    ###################### OPTIMIZER ##########################################
    print('GPU-rank {} : Initializing optimizer...'.format(rank))
    optimizer =  KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = args.momentum, stat_decay = args.stat_decay, 
                                kl_clip = args.kfac_clip, damping = KFAC_damping, 
                                weight_decay = args.WD, TCov = args.TCov_period,
                                TInv = args.TInv_period,
                                # added to deal with efficient work allocation
                                work_alloc_propto_EVD_cost = args.work_alloc_propto_EVD_cost)#   
    print('GPU-rank {} : Done initializing optimizer. Started training...'.format(rank))
    ###################### END: OPTIMIZER #####################################
    
    loss_fn = torch.nn.CrossEntropyLoss() #F.nll_loss #Your_Loss() # nn.CrossEntropyLoss()
    # for test loss use: # nn.CrossEntropyLoss(size_average = False)
    
    ############## schedule function ##########################################
    def schedule_function(optimizer, epoch):
        if epoch in TCov_schedule:
            optimizer.TCov =  TCov_schedule[epoch]
        if epoch in TInv_schedule:
            optimizer.TInv = TInv_schedule[epoch]
        if epoch in KFAC_damping_schedule: 
            optimizer.param_groups[0]['damping'] = KFAC_damping_schedule[epoch]
    ############## END: schedule function #####################################

    ########################## TRAINING LOOP: over epochs ######################################################
    train_n_epochs(model, optimizer, loss_fn, train_set, test_set, schedule_function, args, len_train_set, rank, world_size)
    # how many epochs to train is in args.n_epochs
    ##################### END : TRAINING LOOP: over epochs ####################################################
    
    ####### test at the end of training #####
    if args.test_at_end == True: 
        print('Rank = {}. Testing at the end (i.e. epoch = {})... \n'.format(rank, args.n_epochs + 1))
        test(test_loader = test_set, model = model, loss_fn = loss_fn, rank = rank, world_size = world_size, epoch = args.n_epochs - 1) 
    ## END:  test at the end of training ####
        
    cleanup()
    print('GPU rank = {} of {} is done correctly!'.format(rank, world_size))

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args(solver_name = 'KFAC')
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




