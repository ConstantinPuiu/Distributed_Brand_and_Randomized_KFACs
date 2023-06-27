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

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import partition_dataset, cleanup
from Distributed_Brand_and_Randomized_KFACs.solvers.distributed_R_kfac_lean_Kfactors_batchsize import R_KFACOptimizer
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import l_rate_function
from Distributed_Brand_and_Randomized_KFACs.main_utils.arg_parser_utils import parse_args

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct


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
    batch_size = args.batch_size
    ### RSVD specific params or stuff that 1st appeared in rsvd(clip and damping type)
    rsvd_rank = args.rsvd_rank
    rsvd_oversampling_parameter = args.rsvd_oversampling_parameter
    rsvd_niter = args.rsvd_niter
    damping_type = args.damping_type #'adaptive',
    clip_type = args.clip_type
    
    
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
    rsvd_rank_adaptation_TInv_multiplier = args.rsvd_rank_adaptation_TInv_multiplier
    rsvd_target_truncation_rel_err = args.rsvd_target_truncation_rel_err
    maximum_ever_admissible_rsvd_rank = args.maximum_ever_admissible_rsvd_rank    
    rsvd_adaptive_max_history = args.rsvd_adaptive_max_history
    # ====================================================
    
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
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import TInv_schedule
        if 0 in TInv_schedule.keys(): # overwrite TInv_period
            print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(TInv_period, TInv_schedule[0]))
            TInv_period = TInv_schedule[0]
    
    if args.TCov_schedule_flag == 0: # then it's False
        TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
    else: # if the flag is True
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import TCov_schedule
        if 0 in TCov_schedule.keys(): # overwrite TInv_period
            print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(TCov_period, TCov_schedule[0]))
            TCov_period = TCov_schedule[0]
    
    #########################################
            
    ### for dealing with other parameters SCHEDULES ####
    if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
        KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
    else:
        from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import KFAC_damping_schedule
    KFAC_damping = KFAC_damping_schedule[0]
    ### TO DO: implement the schedules properly: now only sticks at the first entryforever in all 3
    ################################ END SCHEDULES ###################################################################
    
    # ====================================================
        
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
    optimizer =  R_KFACOptimizer(model, rank = rank, world_size = world_size, 
                               lr_function = l_rate_function, momentum = momentum, stat_decay = stat_decay, 
                                kl_clip = kfac_clip, damping = KFAC_damping, 
                                weight_decay = WD, TCov = TCov_period,
                                TInv = TInv_period,
                                rsvd_rank = rsvd_rank,
                                rsvd_oversampling_parameter = rsvd_oversampling_parameter,
                                rsvd_niter = rsvd_niter,
                                work_alloc_propto_RSVD_cost = work_alloc_propto_RSVD_cost,
                                work_eff_alloc_with_time_measurement = work_eff_alloc_with_time_measurement,
                                damping_type = damping_type, #'adaptive',
                                clip_type = clip_type,
                                adaptable_rsvd_rank = adaptable_rsvd_rank,
                                rsvd_target_truncation_rel_err = rsvd_target_truncation_rel_err,
                                maximum_ever_admissible_rsvd_rank = maximum_ever_admissible_rsvd_rank,
                                rsvd_adaptive_max_history = rsvd_adaptive_max_history,
                                rsvd_rank_adaptation_TInv_multiplier = rsvd_rank_adaptation_TInv_multiplier
                                )#    optim.SGD(model.parameters(),
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
            ## debug negative loss only
            #if y not in [0,1,2,3,4,5,6,7,8,9]: # checing for data being corrupted? (it wasn't this it was forgetting the softmax)
            #    print('rank = {} has encountered WEIRD label yi = {}'.format(y))
            
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

if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args(solver_name = 'R-KFAC')
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\nStarted again, Current Time = {} \n for R-KFAC lean\n'.format(now_start))
    print('\nImportant args were:\n  --work_alloc_propto_RSVD_cost = {};\n  --work_eff_alloc_with_time_measurement = {};\n  --adaptable_rsvd_rank = {};\n  --rsvd_rank_adaptation_TInv_multiplier = {}\n'.format(args.work_alloc_propto_RSVD_cost, args.work_eff_alloc_with_time_measurement, args.adaptable_rsvd_rank, args.rsvd_rank_adaptation_TInv_multiplier))
    print('\nScheduling flags were: \n --TInv_schedule_flag = {}, --TCov_schedule_flag = {}, --KFAC_damping_schedule_flag = {}'.format(args.TInv_schedule_flag, args.TCov_schedule_flag, args.KFAC_damping_schedule_flag))
    print('\n !! net_type = {}, dataset = {}'.format(args.net_type, args.dataset))
    
    print('\nDoing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('\nFINISHED AT: = {} \n\n'.format(datetime.now()))





