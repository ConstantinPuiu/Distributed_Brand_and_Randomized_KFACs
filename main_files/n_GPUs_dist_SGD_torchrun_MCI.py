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

from Distributed_Brand_and_Randomized_KFACs.main_utils.data_utils_dist_computing import get_data_loaders_and_s, cleanup
from torch.optim import SGD
from Distributed_Brand_and_Randomized_KFACs.main_utils.lrfct import get_l_rate_function
from Distributed_Brand_and_Randomized_KFACs.main_utils.arg_parser_utils import parse_args, adjust_args_for_0_1_and_compatibility, adjust_args_for_schedules

from Distributed_Brand_and_Randomized_KFACs.main_utils.generic_utils import get_net_main_util_fct, train_n_epochs, test


def main(world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl")#, world_size=world_size)
    rank = dist.get_rank()
    print('Hello from GPU rank {} with pytorch DDP\n'.format(rank))
    
    # adjust args: turn 0-1 variables that should be True / False into True / False AND check and correct (args.net_type, args.dataset) combination
    args = adjust_args_for_0_1_and_compatibility(args, rank, solver_name = 'SGD')
    ###################################### end adjust 0-1 -===> True / False ######################################################################
    
    ################################  SCHEDULES ######################################################################
    args, momentum_dampening_schedule = adjust_args_for_schedules(args, solver_name = 'SGD')
    ################################ END SCHEDULES ###################################################################
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
    lr_schedule_fct = get_l_rate_function(lr_schedule_type = args.lr_schedule_type, base_lr = args.base_lr, 
                                          lr_decay_rate = args.lr_decay_rate, lr_decay_period = args.lr_decay_period, 
                                          auto_scale_forGPUs_and_BS = args. auto_scale_forGPUs_and_BS, n_GPUs = args.world_size,
                                          batch_size = args.batch_size, dataset = args.dataset ) 
    # get_l_rate_function returns fct handle to lr shcedule fct - arguments of the returned function (lr_schedule_fct) is only the epoch index
    
    print('GPU-rank {} : Initializing optimizer...'.format(rank))
    optimizer =  SGD(model.parameters(), lr = args.base_lr,
                     momentum = args.momentum, dampening = args.momentum_dampening, 
                     weight_decay = args.WD, nesterov = args.use_nesterov)#   
    print('GPU-rank {} : Done initializing optimizer. Started training...'.format(rank))
    ###################### END: OPTIMIZER #####################################
    
    loss_fn = torch.nn.CrossEntropyLoss() #F.nll_loss #Your_Loss() # nn.CrossEntropyLoss()
    # for test loss use: # nn.CrossEntropyLoss(size_average = False)
    
    ############## schedule function ##########################################
    def schedule_function(optimizer, epoch):
        ### for SGD only we put the lt scheduler in the schedule function, as the optimizer has no internal check for lr schedule as KFAC which we coded has
        optimizer.lr = lr_schedule_fct(epoch_n = epoch, iter_n = None)
        ### 
        if epoch in momentum_dampening_schedule:
            optimizer.dampening =  momentum_dampening_schedule[epoch]
    # define any scheduler function (apart from lr) that sets schedules working outside the (SGD) optimizer
    ############## END: schedule function #####################################

    ########################## TRAINING LOOP: over epochs ######################################################
    stored_metrics_object = train_n_epochs(model, optimizer, loss_fn, train_loader, train_sampler, val_loader, 
                                           schedule_function, 
                                           args, len_train_loader, rank, world_size, optim_type = 'SGD')
    # how many epochs to train is in args.n_epochs
    ##################### END : TRAINING LOOP: over epochs ####################################################
    
    ####### print and save stored metrics #####################################################################
    if rank == 0 and args.store_and_save_metrics:
        stored_metrics_object.print_metrics()
        stored_metrics_object.get_device_names_and_store_in_object(world_size = args.world_size)
        stored_metrics_object.save_metrics( metrics_save_path = args.metrics_save_path, dataset = args.dataset, 
                                           net_type = args.net_type, solver_name = 'SGD',  nGPUs = args.world_size,
                                           batch_size = args.batch_size, run_seed = args.seed )
    ####### END : print and save stored metrics ###############################################################
        
    cleanup()
    print('GPU rank = {} of {} is done correctly!'.format(rank, world_size) + '\nFINISHED AT: = {} \n\n'.format(datetime.now()))
    print('######### Finished Running SGD at seed = {} ######################################################'.format(args.seed))
    
if __name__ == '__main__':
    # suppose we have 3 gpus
    args = parse_args(solver_name = 'SGD')
    now_start = datetime.now()
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nStarted again, Current Time = {} \n'.format(now_start))
    print('\n ######### Started Running SGD at seed = {} ######################################################'.format(args.seed))
    print('\nStarted again, Current Time = {} \n for SGD \n'.format(now_start))
    print('\nImportant args were:\n --momentum = {} ; \n --WD (weight decay) = {} \n'.format( args.momentum , args.WD ))
    print('\n --batch_size = {} (per GPU for grad, total BS for K-factors); \n --use_nesterov = {} ;\n --momentum_dampening = {}; \n'.format(args.batch_size, args.use_nesterov, args.momentum_dampening ))
    print('\n Learning rate hyper-parameters:\n --lr_schedule_type = {} ; --base_lr = {}; \n --lr_decay_rate = {} ; --lr_decay_period = {}; \n --auto_scale_forGPUs_and_BS = {} \n'.format(args. lr_schedule_type, args.base_lr, args.lr_decay_rate, args.lr_decay_period, args.auto_scale_forGPUs_and_BS) )
    print('\nScheduling flags were: \n --momentum_dampening_schedule_flag = {}\n'.format(args.momentum_dampening_schedule_flag))
    print('\n !! net_type = {}, dataset = {}'.format(args.net_type, args.dataset))
    
    print('\nDoing << {} >> epochs'.format(args.n_epochs))
    world_size = args.world_size
    main(world_size, args)
    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
    #    f.write('\nFINISHED AT: = {} \n\n'.format(datetime.now()))




