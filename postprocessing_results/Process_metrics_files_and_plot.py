import torch
from utils_for_metrics_processor import get_loader_for_solver_only, load_metric_list, get_t_per_epoch_and_step_list,\
    get_t_and_n_ep_list_to_acc, get_mean_and_std, plot_and_save, plot_avg_over_solvers_and_save, check_which_GPU_s

####################### parameters ############################################
root_folder = '/path/to/saved_metrics'

#### VGG16_bn_lmxp CIFAR10 ######################

#date = '2023-07-06' ## cifar10 A100 - 2gpu

#date = '2023-07-11'## cifar10  V100-32GB - 2gpu  # YYYY-MM-DD format
#date = '2023-07-12' ## cifar10  V100-32GB - 4gpu

#### end: VGG16_bn_lmxp CIFAR10 ######################
date = 'CIFAR10'

##### CIFAR100###########################
#date = '2023-07-16'## CIFAR100  V100-32GB - 1GPU, 2GPUs and 4gpus 70% target acc
########################################
""" VGG16_bn_lmxp CIFAR10 Complete results 2 GPUs: A100 @ 2023-07-06; V100-32GB @ 2023-07-11
                                           4 GPUs: A100 @  ; V100-32GB @ 2023-07-12 
"""

solver_list = [ 'SGD', 'KFAC', 'R', 'B', 'BR', 'BRC'] # possible values: R, B, BR, BRC, KFAC, 
net_type = 'VGG16_bn_lmxp' # VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected
dataset = 'cifar10' # 'Possible Choices: MNIST, SVHN, cifar10, cifar100, imagenet, imagenette_fs_v2
batch_size = 128 # batchsize per GPU
num_GPUs = 1 #can be in [ 1, 2, 4 ]

t_acc_criterion = 92.0
savepath = '/path/to/write/plots_from_saved_metrics/'
#################### END parameter ############################################
########### ========= What to do when running ====== ##########################
get_and_print_times = True # gets and prints mean in #runs = len(seed)  and the standard deviation around the mean
plot_convergence_graphs = True
check_GPUs_overview = True
########### ======= END: What to do when running ==== #########################

###################### troch.load data from files #############################
loader_for_solver = get_loader_for_solver_only(net_type, dataset, batch_size, num_GPUs, 
                                               root_folder, date, verbose = True)

all_compressed_metrics_dict = {}

##################################### do plots ################################
if plot_convergence_graphs == True:
    metrics_concatenated_over_solvers = {}
    for solver in solver_list:
        read_metrics_for_solver = loader_for_solver(solver) 
        # test acc
        plot_and_save(read_metrics_for_solver, y_metric = 'test_acc',
                      x_metric = 'epoch_number_test', savepath = savepath, 
                      dataset = dataset,
                      solver_name = solver, horizontal_line_criterion = t_acc_criterion)
        plot_and_save(read_metrics_for_solver, y_metric = 'test_acc',
                      x_metric = 'time_to_epoch_end_test', savepath = savepath, 
                      dataset = dataset,
                      solver_name = solver, horizontal_line_criterion = t_acc_criterion)
        # train acc
        plot_and_save(read_metrics_for_solver, y_metric = 'train_acc', 
                      x_metric = 'epoch_number_train', savepath = savepath, 
                      dataset = dataset,
                      solver_name = solver)
        plot_and_save(read_metrics_for_solver, y_metric = 'train_acc',
                      x_metric = 'time_to_epoch_end_train', savepath = savepath, 
                      dataset = dataset,
                      solver_name = solver)
        # loss
        plot_and_save(read_metrics_for_solver, y_metric = 'train_loss',
                      x_metric = 'epoch_number_train', savepath = savepath, 
                      dataset = dataset,
                      solver_name = solver)
        metrics_concatenated_over_solvers[solver] = read_metrics_for_solver
    
    # average metrics plot
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'test_acc',
                                   x_metric = 'epoch_number_test', dataset = dataset, savepath = savepath,
                                   horizontal_line_criterion = t_acc_criterion)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'test_acc',
                                   x_metric = 'time_to_epoch_end_test', dataset = dataset, savepath = savepath,
                                   horizontal_line_criterion = t_acc_criterion)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'train_acc',
                                   x_metric = 'epoch_number_train', dataset = dataset,  savepath = savepath)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'train_loss',
                                   x_metric = 'epoch_number_train', dataset = dataset,  savepath = savepath)
############################## END: do plots ###################################
    
##################################### do table data ###########################
if get_and_print_times == True:
    print('DOING TABLES!')
    #### save in object
    for solver in solver_list:
        read_metrics_for_solver = loader_for_solver(solver) #format read_metrics[metric][seed]
        
        ## get # epochs and time to epoch  # nsr is number of succesful runs, ntr is number of total runs
        t_to_t_acc_list, n_epoch_to_t_acc_list, nsr, ntr = get_t_and_n_ep_list_to_acc(read_metrics_for_solver, t_acc_criterion)
        m_t_acc, s_t_acc = get_mean_and_std(t_to_t_acc_list)
        m_n_epoch_to_acc, s_n_epoch_to_acc = get_mean_and_std(n_epoch_to_t_acc_list) 
        
        # get time per epoch: this is average, and we should consider the fact that the rank is adaptive, thus changing the time per epoch
        t_per_epoch_list, t_per_step_list, steps_per_epoch = get_t_per_epoch_and_step_list(read_metrics_for_solver, dataset)
        m_t_per_epoch, s_t_per_epoch = get_mean_and_std(t_per_epoch_list)
        
        # get time per step: this is average, and we should consider the fact that the rank is adaptive - so the k-fac worload changes
        m_t_per_step, s_t_per_step = get_mean_and_std(t_per_step_list)
        
        # form dictionaries 
        current_metric_dict = {}
        current_metric_dict['t_to_test_acc'] = m_t_acc, s_t_acc
        current_metric_dict['n_epoch_to_acc'] = m_n_epoch_to_acc, s_n_epoch_to_acc
        current_metric_dict['t_per_epoch'] = m_t_per_epoch, s_t_per_epoch
        current_metric_dict['t_per_step'] = m_t_per_step, s_t_per_step
        current_metric_dict['nsr_and_ntr'] = nsr, ntr
        all_compressed_metrics_dict[solver] = current_metric_dict
    
    # print and / or save
    for solver in solver_list:
        m_t_acc, s_t_acc = all_compressed_metrics_dict[solver]['t_to_test_acc']
        m_n_epoch, s_n_epoch = all_compressed_metrics_dict[solver]['n_epoch_to_acc']
        m_t_per_epoch, s_t_per_epoch = all_compressed_metrics_dict[solver]['t_per_epoch']
        m_t_per_step, s_t_per_step = all_compressed_metrics_dict[solver]['t_per_step']
        nsr, ntr = all_compressed_metrics_dict[solver]['nsr_and_ntr']
        print('Solver: {}\n Time to {:.2f}\% test acc:  {:.2f}+/- {:.2f} s \
              \n N. Epochs to {:.2f}\% test acc:  {:.2f}+/- {:.2f} \
              \n Average Per-epoch time: {:.2f}+/- {:.2f} s \
              \n Average Per-step time: {:.5f}+/- {:.4f} s \
              \n And a [succesful runs] / [total runs]  =  {} / {}\n'.format(
              solver, t_acc_criterion, m_t_acc, s_t_acc,
                      t_acc_criterion, m_n_epoch, s_n_epoch,
                                       m_t_per_epoch, s_t_per_epoch,
                                       m_t_per_step, s_t_per_step,
                                       nsr, ntr
                                       ))
    
    print('\n steps_per_epoch = {}'.format(steps_per_epoch))
            
################################ END: do table data ###########################
    
########### check GPUs easily ########################################
loader_for_solver = get_loader_for_solver_only(net_type, dataset, batch_size, num_GPUs, 
                                               root_folder, date, verbose = False)
if check_GPUs_overview == True:
    ### see if GPUs were the correct ones
    D = {}
    for solver in solver_list:  
        read_metrics_for_solver = loader_for_solver(solver)
        D[solver] = read_metrics_for_solver
        
    GPUs_for_each_solver = check_which_GPU_s(D)
    print('\n=================\n GPUs_for_each_solver = {}\n=================\n'.format(GPUs_for_each_solver))
        
########### END: check GPUs easily ####################################
