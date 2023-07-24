import torch
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import math

##### helper function to:
#### check if code was run on correct GPUs  for all solvers (as desired) 
def check_which_GPU_s(all_compressed_metrics_dict):
        GPUs_for_each_solver = {}
        for solver in all_compressed_metrics_dict.keys():
            GPU_names_for_this_solver = []
            for seed in all_compressed_metrics_dict[solver]['GPU_names'].keys():
                if all_compressed_metrics_dict[solver]['GPU_names'][seed] in GPU_names_for_this_solver:
                    pass
                else:
                    GPU_names_for_this_solver.append(all_compressed_metrics_dict[solver]['GPU_names'][seed])
            GPUs_for_each_solver[solver] = GPU_names_for_this_solver
        return GPUs_for_each_solver

####################
        
############################################################
################### basic loading tools ####################
############################################################

def get_loader_for_solver_only(net_type, dataset, batch_size, num_GPUs, 
                     root_folder, date, verbose = True):
    return (lambda solver: load_metric_list(solver, net_type, dataset, batch_size, num_GPUs, 
                     root_folder, date, verbose = verbose) )

no_training_samples = {'MNIST': 60000, 'SVHN': 530000, 'cifar10': 50000, 
                           'cifar100': 50000, 'imagenet': 1281167, 'imagenette_fs_v2': 9469 }

def load_metric_list(solver, net_type, dataset, batch_size, num_GPUs, 
                     root_folder, date, verbose = True):
    print('Loading data for {} (net_type ={}, \ndataset = {}, \nbatch_size = {}, \nnum_GPUs= {}, \
                     \nroot_folder, \ndate = {}, \nroot_folder = {}\n)...'.format(solver, net_type, dataset, 
                                                    batch_size, num_GPUs, date,
                                                    root_folder     )  )
    
    metric_list = ['epoch_number_train' , 'train_loss' , 'train_acc' , 'time_to_epoch_end_train',
                   'epoch_number_test', 'test_loss' , 'test_acc' , 'time_to_epoch_end_test',
                   'GPU_names']
    
    seed_list = [12345, 23456, 34567, 45678, 56789]
    
    no_steps_per_epoch = math.ceil(no_training_samples[dataset] / num_GPUs / batch_size)
    print('no_steps_per_epoch = {} at dataset = {}, num_GPUs = {}, batch_size = {}'.format(no_steps_per_epoch, dataset, num_GPUs, batch_size))
    
    path = root_folder + '/' + date + '/'
    
    ### initialize read_metrics dict
    read_metrics = {} # index as read_metrics['metric']['seed']
    for metric in metric_list:
        read_metrics[metric] = {}
    
    for seed in seed_list:
        read_metrics['num_GPUs'] = {seed: num_GPUs}
        read_metrics['batch_size'] = {seed: batch_size}
        for metric in metric_list:
            path_final = path + '{}_{}_{}_nGPUs_{}_Bsize_{}_{}_{}.pt'.format(dataset, net_type, solver,
                                                                          num_GPUs, batch_size, metric, seed)
            read_metrics[metric][seed] = torch.load(path_final)
            ### CORRECT FOR WRONG EPOCH STAMP AT FINAL TEST EPOCH WHEN  -##############
            # - this is an artifact of a slight bug in the saving protocol when early stopping occurs
            # (that can be corrected at postprocessing)
            if metric == 'epoch_number_test':
                read_metrics[metric][seed][-1] = read_metrics['epoch_number_train'][seed][-1]
            ### END: CORRECT FOR WRONG EPOCH STAMP AT FINAL TEST EPOCH WHEN  -#########
            
        if verbose == True:
            print('\n ############################################################# \
                  \n seed = {}, dataset = {}, net_type = {}, \n solver = {}, num_GPUs = {}, batch_size = {} \
                  \n has GPU_names = {} \
                  \n ############################################################# \
                  \n'.format(seed, dataset, net_type, solver, num_GPUs, batch_size,
                                                read_metrics['GPU_names'][seed]))
    
    return read_metrics

############################################################
############### END basic loading tools ####################
############################################################
    
##############################################################################################
### Tools for getting table-like presentation ################################################
##############################################################################################

def get_t_per_epoch_and_step_list(read_metrics_for_solver, dataset):
    """ assumed format is read_metrics[metric][seed] for 1 solver
    It is also assumed all metrics have (correctly!) the same number of seeds stored"""
    # returns t_per_epoch_list - actually measured
    # returns t_per_step_list - computed fromt_per_epoch_list using the number of steps per epoch 
    ########### - this is just the average time per step for that epoch
    
    seeds_list = read_metrics_for_solver['epoch_number_train'].keys()
    t_per_epoch_list = []
    t_per_step_list = []
    # steps per epoch stuff
    sedd = list(read_metrics_for_solver['num_GPUs'].keys())[0]
    num_GPUs = read_metrics_for_solver['num_GPUs'][ sedd]
    batch_size = read_metrics_for_solver['batch_size'][ sedd]
    steps_per_epoch = (no_training_samples[dataset] // (num_GPUs * batch_size) ) + 1.0
    # end steps per epoch stuff
    
    for seed in seeds_list:
        current_cumsum_time_list = np.array(read_metrics_for_solver['time_to_epoch_end_train'][seed])
        time_increments_list = list(current_cumsum_time_list[1:] - current_cumsum_time_list[:-1])
        t_per_epoch_list = t_per_epoch_list + time_increments_list
    
    t_per_step_list = list(np.array(t_per_epoch_list) / steps_per_epoch )
    return t_per_epoch_list, t_per_step_list, steps_per_epoch

def get_t_and_n_ep_list_to_acc(read_metrics_for_solver, t_acc_criterion):
    """ assumed format is read_metrics[metric][seed] for 1 solver
    It is also assumed all metrics have (correctly!) the same number of seeds stored"""
    seeds_list = read_metrics_for_solver['epoch_number_train'].keys()
    
    t_to_t_acc_list = []
    n_epoch_to_t_acc_list = []
    for seed in seeds_list:
        time_list = read_metrics_for_solver['time_to_epoch_end_test'][seed]
        epoch_list = read_metrics_for_solver['epoch_number_test'][seed]
        criterion_list = read_metrics_for_solver['test_acc'][seed]
        
        where_indices = np.where(np.array(criterion_list) >= t_acc_criterion)[0]
        print('seed {}. In our criterion-search we got np.where(np.array(criterion_list) >= t_acc_criterion) ={}\
              \n But we are saving the 1st one as it is the first time it hits the criterion!!\n'.format(seed, where_indices) )
        idx_first_criterion_true = where_indices[0]
        t_to_t_acc = time_list[idx_first_criterion_true]
        n_epoch_to_t_acc = epoch_list[idx_first_criterion_true]
        #append to list
        t_to_t_acc_list.append(t_to_t_acc)
        n_epoch_to_t_acc_list.append(n_epoch_to_t_acc)
    
    return t_to_t_acc_list, n_epoch_to_t_acc_list


def get_mean_and_std(data_list):
    return np.mean(data_list), np.std(data_list)
##############################################################################################
##############################################################################################

###############################################
#### tools for getting plots ##################
###############################################
thesis_names_dict = {'epoch_number_train': 'Epochs', 'train_loss' : 'Train Loss', 'train_acc' : 'Train Accuracy', 
                     'time_to_epoch_end_train' : 'Time',
                   'epoch_number_test' : 'Epochs', 'test_loss' : 'Test Loss', 'test_acc' : 'Test Accuracy',
                     'time_to_epoch_end_test': 'Time',
                     'SGD': 'SGD', 'KFAC': 'K-FAC', 'R': 'R-KFAC', 
                      'B': 'B-KFAC',  'BR': 'BR-KFAC',  'BRC': 'BRC-KFAC'}
units_dict = {'Time': ' s', 'Epochs': '', 'Train Accuracy': ' %', 'Test Accuracy': ' %', 'Train Loss': '', 'Test Loss': ''}

def check_make_path(path):
    if os.path.exists(path):
        path# if the path already exists then do nothing
    else: # if it doesnt exist, create it
        os.mkdir(path)
    return path

def plot_and_save(metrics_for_1_solver, y_metric, x_metric, dataset,  savepath = None, solver_name = None):
    # assumed format is read_metrics[metric][seed] for 1 solver
    if savepath is None:
        print('Not saving plot y_metric = {}, x_metric = {} '.format(y_metric, x_metric))
    if solver_name is None:
        raise ValueError('solver_name passed to plot_and_save cannot be None !')
    
    sedd = list(metrics_for_1_solver['num_GPUs'].keys())[0]
    num_GPUs = metrics_for_1_solver['num_GPUs'][ sedd]
    ### assuming all GPUs are the same - nned to check manually r introduce automatic check
    GPU_name = metrics_for_1_solver['GPU_names'][ sedd]
    GPU_name = GPU_name[0]
    GPU_spec_for_title = str(num_GPUs) + ' x ' + GPU_name
    
    y_dict = metrics_for_1_solver[y_metric]
    x_dict = metrics_for_1_solver[x_metric]
    plt.figure(figsize = (8, 6))
    plt.title('{}: {} vs {} \n on {} GPUs'.format(thesis_names_dict[solver_name], thesis_names_dict[y_metric], 
                                          thesis_names_dict[x_metric], GPU_spec_for_title) )
    plt.xlabel('{}{}'.format(thesis_names_dict[x_metric], units_dict[thesis_names_dict[x_metric]]))
    plt.ylabel('{}{}'.format(thesis_names_dict[y_metric], units_dict[thesis_names_dict[x_metric]]))
    # plt.hold( True ) # deprecated behaviour past v 2.1, auto set to true, can't se tto false
    
    for idx, seed in enumerate(y_dict.keys()):
        y_axis = y_dict[seed]
        x_axis = x_dict[seed]
        if thesis_names_dict[y_metric] in ['Train Loss', 'Test Loss']:
            plt.semilogy(x_axis, y_axis, label = 'Run {}'.format(idx))
        else:
            plt.plot(x_axis, y_axis, label = 'Run {}'.format(idx))
    plt.legend()
    
    if savepath is None:
        plt.show()
    else:
        savepath = check_make_path(savepath + '{}/'.format(dataset))
        savepath = check_make_path(savepath + '{}_{}/'.format( solver_name, GPU_spec_for_title))
        plt.savefig(savepath + '{}_{}_vs_{}.eps'.format(solver_name, y_metric, thesis_names_dict[x_metric]), format = 'eps')
        plt.savefig(savepath + '{}_{}_vs_{}.png'.format(solver_name, y_metric, thesis_names_dict[x_metric]), format = 'png')
    
    
def plot_avg_over_solvers_and_save(metrics_concatenated_over_solvers, y_metric, \
                                   x_metric, dataset, savepath = None):
    if savepath is None:
        print('Not saving plot y_metric = {}, x_metric = {} '.format(y_metric, x_metric))
    
    #### 1. average over solver , for each solver ############
    averaged_over_solver_x_axis = {}
    averaged_over_solver_y_axis = {}
    
    if thesis_names_dict[x_metric] not in ['Epochs']:
        x_averageing_required = True
    else:
        x_averageing_required = False
    
    ############ ASSUMING ALL SOLVERS AND ALL RUNS ON SAME GPU, ELSE NEED TO CHECK MANUALLY - INTRODUCE AUTO CHECK
    print(metrics_concatenated_over_solvers.keys())
    slv = list(metrics_concatenated_over_solvers.keys())[0]
    sedd = list(metrics_concatenated_over_solvers[slv]['num_GPUs'].keys())[0]
    num_GPUs = metrics_concatenated_over_solvers[slv]['num_GPUs'][sedd]
    ### assuming all GPUs are the same - nned to check manually r introduce automatic check
    GPU_name = metrics_concatenated_over_solvers[slv]['GPU_names'][sedd]
    GPU_name = GPU_name[0]
    GPU_spec_for_title = str(num_GPUs) + ' x ' + GPU_name
    
    for solver in metrics_concatenated_over_solvers.keys():
        unaveraged_list_of_x = []
        unaveraged_list_of_y = []
        # for over seeds
        for seed in metrics_concatenated_over_solvers[solver][x_metric].keys():
            unaveraged_list_of_x.append( metrics_concatenated_over_solvers[solver][x_metric][seed] )
            unaveraged_list_of_y.append( metrics_concatenated_over_solvers[solver][y_metric][seed] )
        
        #### do the avrerage
        def get_average_over_list_with_early_stopping(unaveraged_list_of_lists):
            min_len = 1e16
            # get min length
            for lisst in unaveraged_list_of_lists:
                if len(lisst) < min_len:
                    min_len = len(lisst)
            # correct lists to min length
            for idx in range(0,len(unaveraged_list_of_lists)):
                unaveraged_list_of_lists[idx] = unaveraged_list_of_lists[idx][0:min_len]
            
            return list( np.average(np.array(unaveraged_list_of_lists), axis = 0) )
        
        averaged_over_solver_y_axis[solver] = get_average_over_list_with_early_stopping(unaveraged_list_of_y)
        
        if x_averageing_required == True:
             averaged_over_solver_x_axis[solver] = get_average_over_list_with_early_stopping(unaveraged_list_of_x)#
        else: # even if averaging is not required, still do it to ensure length of list is right 
            #(as some runs take less epochs than others and the avg function we call solves this issue under the hood)
            averaged_over_solver_x_axis[solver] = get_average_over_list_with_early_stopping(unaveraged_list_of_x)
        
    # 2. plot
    plt.figure(figsize = (8, 6))
    plt.title('All solvers: {} vs {} \n on {} GPUs'.format(thesis_names_dict[y_metric], 
                                                        thesis_names_dict[x_metric], GPU_spec_for_title) )
    if thesis_names_dict[x_metric] not in ['Epochs']:
        average_qmark = 'Average '
    else:
        average_qmark = ''
        
    plt.xlabel('{}{}{}'.format(average_qmark, thesis_names_dict[x_metric], units_dict[thesis_names_dict[x_metric]]))
    plt.ylabel('{}{}{}'.format(average_qmark, thesis_names_dict[y_metric], units_dict[thesis_names_dict[y_metric]]))
    # plt.hold( True ) # deprecated behaviour past v 2.1, auto set to true, can't se tto false
    
    for solver in metrics_concatenated_over_solvers.keys():
        y_axis = averaged_over_solver_y_axis[solver]
        x_axis = averaged_over_solver_x_axis[solver]
        if thesis_names_dict[y_metric] in ['Train Loss', 'Test Loss']:
            plt.semilogy(x_axis, y_axis, label = '{}'.format(thesis_names_dict[solver]))
        else:
            try:
                plt.plot(x_axis, y_axis, label = '{}'.format(thesis_names_dict[solver]))
            except:
                pass
        
    plt.legend()
    
    if savepath is None:
        plt.show()
    else:
        savepath = check_make_path(savepath + '{}/'.format(dataset))
        savepath = check_make_path(savepath + '{}_{}/'.format('All_solvers', GPU_spec_for_title))
        plt.savefig(savepath + '{}_{}_vs_{}.eps'.format('allsolvers_avgplots_', y_metric, thesis_names_dict[x_metric]), format = 'eps')
        plt.savefig(savepath + '{}_{}_vs_{}.png'.format('allsolvers_avgplots_', y_metric, thesis_names_dict[x_metric]), format = 'png')
    
###############################################
###############################################
###############################################