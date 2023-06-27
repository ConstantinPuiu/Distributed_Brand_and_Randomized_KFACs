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

#from true_kfac_FC_project_adaptive_damping import KFACOptimizer #distributed_kfac_simplest_form
from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import partition_dataset, cleanup
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_B_kfac_lean_Kfactors_batchsize import B_KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import l_rate_function
from Distributed_Brand_and_Randomized_KFACs.main_utils.arg_parser_utils import parse_args, adjust_args_for_0_1_and_compatibility

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct


def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['NCCL_BLOCKING_WAIT'] = '1'
    #os.environ['NCCL_LL_THRESHOLD'] = '0'
    dist.init_process_group(backend = "nccl", timeout = dateT.timedelta(seconds = 120))#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    # adjust args: turn 0-1 variables that should be True / False into True / False AND check and correct (args.net_type, args.dataset) combination
    args = adjust_args_for_0_1_and_compatibility(args, rank, solver_name = 'B-KFAC')
    
    ###others only added after starting CIFAR10
    TCov_period = args.TCov_period
    TInv_period = args.TInv_period
    
    ######### BRAND K-fac (also BRSKFAC) specific parameters
    #brand_period = args.brand_period 
    brand_r_target_excess = args.brand_r_target_excess
    brand_update_multiplier_to_TCov = args.brand_update_multiplier_to_TCov
    # ====================================================
    
    rsvd_rank_adaptation_TInv_multiplier = args.rsvd_rank_adaptation_TInv_multiplier
    rsvd_target_truncation_rel_err = args.rsvd_target_truncation_rel_err
    maximum_ever_admissible_rsvd_rank = args.maximum_ever_admissible_rsvd_rank    
    rsvd_adaptive_max_history = args.rsvd_adaptive_max_history
    # ====================================================
    
    B_rank_adaptation_T_brand_updt_multiplier = args.B_rank_adaptation_T_brand_updt_multiplier
    B_target_truncation_rel_err = args.B_target_truncation_rel_err
    maximum_ever_admissible_B_rank = args.maximum_ever_admissible_B_rank    
    B_adaptive_max_history = args.B_adaptive_max_history
    # ===================================================
    
    # ====================================================
    
    ################################  SCHEDULES ######################################################################
    ### for dealing with PERIOD SCHEDULES
    if args.TInv_schedule_flag == 0: # then it's False
        TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
    else:# if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import TInv_schedule
        if 0 in TInv_schedule.keys(): # overwrite TInv_period
            print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(TInv_period, TInv_schedule[0]))
            TInv_period = TInv_schedule[0]
    
    if args.TCov_schedule_flag == 0: # then it's False
        TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
    else: # if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import TCov_schedule
        if 0 in TCov_schedule.keys(): # overwrite TInv_period
            print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(TCov_period, TCov_schedule[0]))
            TCov_period = TCov_schedule[0]
    
    if args.brand_update_multiplier_to_TCov_schedule_flag == 0: # then it's False
        brand_update_multiplier_to_TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
    else: # if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import brand_update_multiplier_to_TCov_schedule
        if 0 in brand_update_multiplier_to_TCov_schedule.keys(): # overwrite TInv_period
            print('Because --brand_update_multiplier_to_TCov_schedule_flag was set to non-zero (True) and brand_update_multiplier_to_TCov_schedule[0] exists, we overwrite brand_update_multiplier_to_TCov = {} (as passed in --brand_update_multiplier_to_TCov) to brand_update_multiplier_to_TCov_schedule[0] = {}'.format(brand_update_multiplier_to_TCov, brand_update_multiplier_to_TCov_schedule[0]))
            brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov_schedule[0]
    
    #########################################
            
    ### for dealing with other parameters SCHEDULES ####
    if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
        KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
    else:
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import KFAC_damping_schedule
    KFAC_damping = KFAC_damping_schedule[0]
    ### TO DO: implement the schedules properly: now only sticks at the first entryforever in all 3
    ################################ END SCHEDULES ###################################################################
    
    # ====================================================###############################
    
    def collation_fct(x):
        return  tuple(x_.to(torch.device('cuda:{}'.format(rank))) for x_ in default_collate(x))

    print('GPU-rank {} : Partitioning dataset ...'.format(rank))
    t_partition_dset_1 = time.time()
    train_set, testset, bsz, num_classes = partition_dataset(collation_fct, args.data_root_path, args.dataset, args.batch_size)
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

    print('GPU-rank {} : Initializing optimizer...'.format(rank))
    optimizer =  B_KFACOptimizer(model, rank = rank, world_size = world_size, 
                                lr_function = l_rate_function, momentum = args.momentum, stat_decay = args.stat_decay, 
                                kl_clip = args.kfac_clip, damping = KFAC_damping, 
                                weight_decay = args.WD, TCov = TCov_period,
                                TInv = TInv_period,
                                rsvd_rank = args.rsvd_rank,
                                rsvd_oversampling_parameter = args.rsvd_oversampling_parameter,
                                rsvd_niter = args.rsvd_niter,
                                damping_type = args.damping_type, #'adaptive',
                                clip_type = args.clip_type,
                                brand_r_target_excess = brand_r_target_excess,
                                brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov,
                                #added to dea with truncation before inversion
                                B_truncate_before_inversion = args.B_truncate_before_inversion,
                                # added to deal with eff work alloc
                                work_alloc_propto_RSVD_and_B_cost = args.work_alloc_propto_RSVD_and_B_cost,
                                # for dealing with adaptable rsvd rank
                                adaptable_rsvd_rank = args.adaptable_rsvd_rank,
                                rsvd_target_truncation_rel_err = rsvd_target_truncation_rel_err,
                                maximum_ever_admissible_rsvd_rank = maximum_ever_admissible_rsvd_rank,
                                rsvd_adaptive_max_history = rsvd_adaptive_max_history,
                                rsvd_rank_adaptation_TInv_multiplier = rsvd_rank_adaptation_TInv_multiplier,
                                # for dealing with adaptable B rank
                                adaptable_B_rank = args.adaptable_B_rank,
                                B_rank_adaptation_T_brand_updt_multiplier = B_rank_adaptation_T_brand_updt_multiplier,
                                B_target_truncation_rel_err = B_target_truncation_rel_err,
                                maximum_ever_admissible_B_rank = maximum_ever_admissible_B_rank,    
                                B_adaptive_max_history = B_adaptive_max_history
                                )#    optim.SGD(model.parameters(),
                              #lr=0.01, momentum=0.5) #Your_Optimizer()
    loss_fn = torch.nn.CrossEntropyLoss() #F.nll_loss #Your_Loss() # nn.CrossEntropyLoss()
    # for test loss use: # nn.CrossEntropyLoss(size_average = False)
    print('GPU-rank {} : Done initializing optimizer. Started training...'.format(rank))
    
    tstart = time.time()
    for epoch in range(0, args.n_epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        #dataloader.sampler.set_epoch(epoch)
        
        ######### setting parameters according to SCHEDULES ###################
        if epoch in TCov_schedule:
            optimizer.TCov =  TCov_schedule[epoch]
        if epoch in TInv_schedule:
            optimizer.TInv = TInv_schedule[epoch]
        if epoch in brand_update_multiplier_to_TCov_schedule:
            optimizer.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov_schedule[epoch]
            optimizer.T_brand_updt = optimizer.TCov * optimizer.brand_update_multiplier_to_TCov # this line is crucial, as we use optimizer.T_brand_updt most of the times!
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
            if jdx == len_train_set - 1 and epoch == args.n_epochs - 1:
                tend = time.time()
                print('TIME: {:.3f} s. Rank (GPU number) {} at batch {}, total steps optimizer.steps = {}:'.format(tend-tstart, rank, jdx, optimizer.steps) + ', epoch ' +str(epoch + 1) + ', loss: {}\n'.format(str(loss.item())))
                #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
                #    f.write('Rank (GPU number) {} at batch {}:'.format(rank, jdx) + str(dist.get_rank())+ ', epoch ' +str(epoch+1) + ', loss: {}\n'.format(str(loss.item())))
            optimizer.step(epoch_number = epoch + 1, error_savepath = None)
    cleanup()
    print('GPU rank = {} of {} is done correctly!'.format(rank, world_size))

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args(solver_name = 'B-KFAC')
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for B-pure-KFAC lean with brand_r_target_excess = {}, brand_update_multiplier_to_TCov = {}\n'.format(now_start, args.brand_r_target_excess, args.brand_update_multiplier_to_TCov))
    print('\nImportant args were:\n  --work_alloc_propto_RSVD_and_B_cost = {} ; \n--B_truncate_before_inversion = {}; \n--adaptable_rsvd_rank = {}; \n--rsvd_rank_adaptation_TInv_multiplier = {};\n --adaptable_B_rank = {}; \n --B_rank_adaptation_T_brand_updt_multiplier = {};\n'.format(args.work_alloc_propto_RSVD_and_B_cost, args.B_truncate_before_inversion, args.adaptable_rsvd_rank, args.rsvd_rank_adaptation_TInv_multiplier,args.adaptable_B_rank, args.B_rank_adaptation_T_brand_updt_multiplier))
    print('\nScheduling flags were: \n --TInv_schedule_flag = {}, --TCov_schedule_flag = {},\n --brand_update_multiplier_to_TCov_schedule_flag = {}, \n--KFAC_damping_schedule_flag = {}'.format(args.TInv_schedule_flag, args.TCov_schedule_flag, args.brand_update_multiplier_to_TCov_schedule_flag, args.KFAC_damping_schedule_flag))
    print('\n !! net_type = {}, dataset = {}'.format(args.net_type, args.dataset))
    #print('type of brand_r_target_excess is {}'.format(type(args.brand_r_target_excess)))
    print('\nDoing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))




