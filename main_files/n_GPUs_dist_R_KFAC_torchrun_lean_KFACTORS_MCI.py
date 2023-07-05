import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import time
from datetime import datetime
import datetime as dateT
from torch.utils.data.dataloader import default_collate

print('torch.__version__ = {}'.format(torch.__version__))

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import get_data_loaders_and_s, cleanup
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_R_kfac_lean_Kfactors_batchsize import R_KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import get_l_rate_function_for_dataset
from Distributed_Brand_and_Randomized_KFACs.main_utils.arg_parser_utils import parse_args, adjust_args_for_0_1_and_compatibility, adjust_args_for_schedules

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct, train_n_epochs, test


def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", timeout = dateT.timedelta(seconds = 120))#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    # adjust args: turn 0-1 variables that should be True / False into True / False AND check and correct (args.net_type, args.dataset) combination
    args = adjust_args_for_0_1_and_compatibility(args, rank, solver_name = 'R-KFAC')
    ###################################### end adjust 0-1 -===> True / False ######################################################################
    
    ################################  SCHEDULES ######################################################################
    args, TInv_schedule, TCov_schedule, KFAC_damping_schedule, KFAC_damping = adjust_args_for_schedules(args, solver_name = 'R-KFAC')
    ################################ END SCHEDULES ###################################################################
    
    # ====================================================
        
    ####
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))
    
    print('GPU-rank {} : Building DataLoaders and DataSamplers (Partitioning dataset) ...'.format(rank))
    t_partition_dset_1 = time.time()
    
    ############################ Get data loaders and samplers ################################
    train_sampler, train_loader, _, val_loader, \
        batch_size, num_classes = get_data_loaders_and_s(args.data_root_path, args.dataset,
                                                         args.batch_size, seed = args.seed)
    # NOTE: seeding of random, torch, and torch.cuda is done inside partition_dataset() call
    ####################### END : Get data loaders and samplers ################################
    
    len_train_loader = len(train_loader)
    t_partition_dset_2 = time.time()
    print('GPU-rank {} : Done building DataLoaders and DataSamplers in {:.2f} s! : len(train_loader) = {}'.format(rank, t_partition_dset_2 - t_partition_dset_1, len_train_loader))
    
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
    optimizer =  R_KFACOptimizer(model, rank = rank, world_size = world_size, batch_size = args.batch_size,
                               lr_function = get_l_rate_function_for_dataset(args.dataset), 
                               momentum = args.momentum, stat_decay = args.stat_decay, 
                                kl_clip = args.kfac_clip, damping = KFAC_damping, 
                                weight_decay = args.WD, TCov = args.TCov_period,
                                TInv = args.TInv_period,
                                damping_type = args.damping_type, #'adaptive',
                                clip_type = args.clip_type,
                                # specific to R-K-fac
                                rsvd_rank = args.rsvd_rank,
                                rsvd_oversampling_parameter = args.rsvd_oversampling_parameter,
                                rsvd_niter = args.rsvd_niter,
                                # added to deal with eff work alloc
                                work_alloc_propto_RSVD_cost = args.work_alloc_propto_RSVD_cost,
                                work_eff_alloc_with_time_measurement = args.work_eff_alloc_with_time_measurement,
                                # added to deal with adaptable rsvd rank
                                adaptable_rsvd_rank = args.adaptable_rsvd_rank,
                                rsvd_rank_adaptation_TInv_multiplier = args.rsvd_rank_adaptation_TInv_multiplier,
                                rsvd_target_truncation_rel_err = args.rsvd_target_truncation_rel_err,
                                maximum_ever_admissible_rsvd_rank = args.maximum_ever_admissible_rsvd_rank, 
                                rsvd_adaptive_max_history = args.rsvd_adaptive_max_history
                                )#    
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
    stored_metrics_object = train_n_epochs(model, optimizer, loss_fn, train_loader, train_sampler, val_loader, 
                                           schedule_function, 
                                           args, len_train_loader, rank, world_size)
    # how many epochs to train is in args.n_epochs
    ##################### END : TRAINING LOOP: over epochs ####################################################
    
   ####### print and save stored metrics #####################################################################
    if rank == 0 and args.store_and_save_metrics:
        stored_metrics_object.print_metrics()
        stored_metrics_object.get_device_names_and_store_in_object(world_size = args.world_size)
        stored_metrics_object.save_metrics( metrics_save_path = args.metrics_save_path, dataset = args.dataset, 
                                           net_type = args.net_type, solver_name = 'R',  nGPUs = args.world_size,
                                           batch_size = args.batch_size, run_seed = args.seed )
    ####### END : print and save stored metrics ###############################################################
    
    cleanup()
    print('GPU rank = {} of {} is done correctly!'.format(rank, world_size) + '\nFINISHED AT: = {} \n\n'.format(datetime.now()))

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args(solver_name = 'R-KFAC')
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for R-KFAC lean\n'.format(now_start))
    print('\nImportant args were:\n  --work_alloc_propto_RSVD_cost = {};\n  --work_eff_alloc_with_time_measurement = {};\n  --adaptable_rsvd_rank = {};\n  --rsvd_rank_adaptation_TInv_multiplier = {}\n'.format(args.work_alloc_propto_RSVD_cost, args.work_eff_alloc_with_time_measurement, args.adaptable_rsvd_rank, args.rsvd_rank_adaptation_TInv_multiplier))
    print('\n--batch_size = {} (per GPU for grad, total BS for K-factors); \n --TInv_period = {} ;\n -- TCov_period = {}; \n'.format(args.batch_size, args.TInv_period, args.TCov_period ))
    print('\nScheduling flags were: \n --TInv_schedule_flag = {}, --TCov_schedule_flag = {}, --KFAC_damping_schedule_flag = {}'.format(args.TInv_schedule_flag, args.TCov_schedule_flag, args.KFAC_damping_schedule_flag))
    print('\n !! net_type = {}, dataset = {}'.format(args.net_type, args.dataset))
    
    print('\nDoing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))





