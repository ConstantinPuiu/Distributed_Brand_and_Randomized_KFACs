import torch
from utils_for_metrics_processor import get_loader_for_solver_only, load_metric_list, get_t_per_epoch_list,\
    get_t_and_n_ep_list_to_acc, get_mean_and_std, plot_and_save, plot_avg_over_solvers_and_save

####################### parameters ############################################
root_folder = 'path_to_where_metrics_are_saved'
date = '2023-07-13' # YYYY-MM-DD format

solver_list = ['SGD', 'KFAC', 'R', 'B', 'BR', 'BRC'] # possible values: R, B, BR, BRC, KFAC, SGD
net_type = 'VGG16_bn_lmxp' # VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected
dataset = 'cifar10' # 'Possible Choices: MNIST, SVHN, cifar10, cifar100, imagenet, imagenette_fs_v2
batch_size = 128 # batchsize per GPU
num_GPUs = 2 #can be in [ 1, 2, 4 ]

t_acc_criterion = 92.0
savepath = 'path_where_to_save_plots'
#################### END parameter ############################################
########### ========= What to do when running ====== ##########################
get_and_print_times = True # gets and prints mean in #runs = len(seed)  and the standard deviation around the mean
plot_convergence_graphs = True
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
                      solver_name = solver)
        plot_and_save(read_metrics_for_solver, y_metric = 'test_acc',
                      x_metric = 'time_to_epoch_end_test', savepath = savepath, 
                      solver_name = solver)
        # train acc
        plot_and_save(read_metrics_for_solver, y_metric = 'train_acc', 
                      x_metric = 'epoch_number_train', savepath = savepath, 
                      solver_name = solver)
        plot_and_save(read_metrics_for_solver, y_metric = 'train_acc',
                      x_metric = 'time_to_epoch_end_train', savepath = savepath, 
                      solver_name = solver)
        # loss
        plot_and_save(read_metrics_for_solver, y_metric = 'train_loss',
                      x_metric = 'epoch_number_train', savepath = savepath, 
                      solver_name = solver)
        metrics_concatenated_over_solvers[solver] = read_metrics_for_solver
    
    # average metrics plot
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'test_acc',
                                   x_metric = 'epoch_number_test', savepath = savepath)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'test_acc',
                                   x_metric = 'time_to_epoch_end_test', savepath = savepath)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'train_acc',
                                   x_metric = 'epoch_number_train', savepath = savepath)
    plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric = 'train_loss',
                                   x_metric = 'epoch_number_train', savepath = savepath)
############################## END: do plots ###################################
    
##################################### do table data ###########################
if get_and_print_times == True:
    #### save in object
    for solver in solver_list:
        read_metrics_for_solver = loader_for_solver(solver) #format read_metrics[metric][seed]
        
        ## get # epochs and time to epoch 
        t_to_t_acc_list, n_epoch_to_t_acc_list = get_t_and_n_ep_list_to_acc(read_metrics_for_solver, t_acc_criterion)
        m_t_acc, s_t_acc = get_mean_and_std(t_to_t_acc_list)
        m_n_epoch_to_acc, s_n_epoch_to_acc = get_mean_and_std(n_epoch_to_t_acc_list) 
        
        # get time per epoch: this is average, and we should consider the fact that the rank is adaptive, thus changing the time per epoch
        t_per_epoch_list = get_t_per_epoch_list(read_metrics_for_solver)
        m_t_per_epoch, s_t_per_epoch = get_mean_and_std(t_per_epoch_list)
        
        # form dictionaries 
        current_metric_dict = {}
        current_metric_dict['t_to_test_acc'] = m_t_acc, s_t_acc
        current_metric_dict['n_epoch_to_acc'] = m_n_epoch_to_acc, s_n_epoch_to_acc
        current_metric_dict['t_per_epoch'] = m_t_per_epoch, s_t_per_epoch
        all_compressed_metrics_dict[solver] = current_metric_dict
        
    # print and / or save
    for solver in solver_list:
        m_t_acc, s_t_acc = all_compressed_metrics_dict[solver]['t_to_test_acc']
        m_n_epoch, s_n_epoch = all_compressed_metrics_dict[solver]['n_epoch_to_acc']
        m_t_per_epoch, s_t_per_epoch = all_compressed_metrics_dict[solver]['t_per_epoch']
        print('Solver: {}\n Time to {:.2f}\% test acc:  {:.2f}+/- {:.2f} s \
              \n N. Epochs to {:.2f}\% test acc:  {:.2f}+/- {:.2f} \
              \n Average Per-epoch time: {:.2f}+/- {:.2f}'.format(
              solver, t_acc_criterion, m_t_acc, s_t_acc,
                      t_acc_criterion, m_n_epoch, s_n_epoch,
                                       m_n_epoch, s_n_epoch))
################################ END: do table data ###########################
