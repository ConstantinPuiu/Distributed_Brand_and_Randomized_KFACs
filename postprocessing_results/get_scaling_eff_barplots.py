import pickle
import matplotlib.pyplot as plt
import numpy as np

savepath = '/'

net_type = 'VGG16_bn_lmxp' # VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected
dataset = 'cifar10' # 'Possible Choices: MNIST, SVHN, cifar10, cifar100, imagenet, imagenette_fs_v2
batch_size = 128 # batchsize per GPU
num_GPUs_list = [1, 2, 4]

criterion_dict = {'cifar10': 92.0, 'cifar100': 70.0, 'SVHN':95.0, 'Imagenette_fs_v2': 92.0}
dict_metric_to_title = {'t_to_test_acc': 'Time to {}% Test Accuracy'.format(criterion_dict[dataset]),
                        'n_epoch_to_acc': '#Epochs to {}% Test Accuracy'.format(criterion_dict[dataset]),
                        't_per_epoch': 'Time Per Epoch',
                        't_per_step': 'Time Per Step',
                        'nsr_and_ntr' : 'Number of Successful Runs'}
dict_net_to_title = {'VGG16_bn_lmxp': 'VGG16-bn-lmxp'}
dict_dataset_to_title = {'cifar10' : 'CIFAR10', 'cifar100' : 'CIFAR100', 'SVHN':'SVHN'}
colordict = {0:'grey', 1:'darkorange', 2:'green', 3:'lightblue', 4:'purple', 5:'brown'}
additive_factor_for_bar_numbers_dict = {'t_to_test_acc': 433,
                                        'n_epoch_to_acc': 10,
                                        't_per_epoch': 20,
                                        't_per_step': 0.05}
solver_list = [ 'SGD', 'KFAC', 'R', 'B', 'BR', 'BRC']

std_multipl_factor_onPlot = 3
figsize = (12 , 2.3 * 5)
ncol_legend = 2
barWidth = 0.155
Metrics_to_plot_list = ['t_to_test_acc', 'n_epoch_to_acc', 't_per_epoch', 't_per_step', 'nsr_and_ntr'] # format Metrics_dict[metric][solver][mean//std]

############ create object that combines over GPUs ############################
# after comining, the dictionary will be D[num_GPUs][solver][metric] = (mean, std); exception: nsr, ntr for nsr_ntr metric
largest_dict_data = {}
for num_GPUs in num_GPUs_list:
    t_acc_criterion = criterion_dict[dataset]
    
    with open('{}_{}_nGPUs_{}_crit_{}_bs_{}.pkl'.format(dataset, net_type, num_GPUs, t_acc_criterion, batch_size), 'rb') as fp:
        current_dict = pickle.load(fp)
    
    largest_dict_data[num_GPUs] = current_dict

############ END: create object that combines over GPUs #######################

def get_ylabel(metric):
    if ('t_to' in metric) or ('t_per' in metric):
        return 'Time (s)'
    elif 'n_epoch' in metric:
        return '# Epochs'
    elif metric == 'nsr_and_ntr':
        return '# Succ. Runs'
    else:
        return ''
    
def get_eff(mean, metric):
    if metric == 't_to_test_acc':
        return mean[0] / ( np.array(num_GPUs_list) * np.array(mean))
    elif metric == 'n_epoch_to_acc': 
        return mean[0] / ( np.array(mean))
    elif metric == 't_per_epoch': 
        return mean[0] / ( np.array(num_GPUs_list) * np.array(mean))
    elif metric == 't_per_step': 
        return mean[0] / ( np.array(mean))
    else:
        raise ValueError('Eff for metric  == {} not defined'.format(metric))
############ iterate over dict object to create barplot #######################

plt.figure(figsize = figsize)
plt.subplot(5, 1, 1)
for imetr, metric in enumerate (Metrics_to_plot_list):
    plt.subplot(5, 1, imetr + 1)
    ymax = 0
    for idx_s, solver in enumerate(solver_list):
        mean_for_current_metric_current_solver = []
        std_for_current_metric_current_solver = []
        nsr_for_current_metric_current_solver = []
        ntr_for_current_metric_current_solver = []
        for num_GPUs in num_GPUs_list:
            mean, std = largest_dict_data[num_GPUs][solver][metric]
            
            mean_for_current_metric_current_solver.append(mean)
            std_for_current_metric_current_solver.append(std)
        
        r1 = np.arange(len(mean_for_current_metric_current_solver)) + idx_s * barWidth
        #### generate xaxis ticks to save space by not labelling xaxis
        r1_ticks = []
        for idx in range(0, len(r1)):
            r1_ticks.append(str(num_GPUs_list[idx]) + ' GPUs')
            
        plt.bar(r1, mean_for_current_metric_current_solver, 
                         #yerr = std_for_current_metric_current_solver,
                         width=barWidth, edgecolor='white', color = colordict[idx_s],
                         label=solver)
        if metric == 'nsr_and_ntr': # in this case nsr = mean, ntr = std
            nsr_list = mean_for_current_metric_current_solver
            ntr_list = std_for_current_metric_current_solver
            ymax = ntr_list[0] * 1.3/ 1.15
            for x_idx in range(0,len(r1)):
                plt.text(r1[x_idx] -0*  barWidth/2.0, 
                         nsr_list[x_idx] / 2,
                         '{}'.format(nsr_list[x_idx]), color = 'white',
                         ha = 'center', fontsize = 'x-large', fontweight = 700,
                         #Bbox = dict(facecolor = colordict[idx_s], alpha =.4)
                         )
        else:
            plt.errorbar(r1, mean_for_current_metric_current_solver, 
                         yerr = std_multipl_factor_onPlot*np.array(std_for_current_metric_current_solver),
                         fmt=".", color="k",
                         #uplims = True, lolims = True,
                         capsize=4, capthick=2)
            
            ### get efficiency for certain metric
            eff_vec = get_eff(mean_for_current_metric_current_solver, metric)
            ### add scaling efficiencies
            for x_idx in range(0,len(r1)):
                y_to_place_eff_label_at = mean_for_current_metric_current_solver[x_idx] + additive_factor_for_bar_numbers_dict[metric] + std_multipl_factor_onPlot*np.array(std_for_current_metric_current_solver)[x_idx]
                if ymax < y_to_place_eff_label_at:
                    ymax = y_to_place_eff_label_at
                plt.text(r1[x_idx] -0*  barWidth/2.0, 
                         y_to_place_eff_label_at,
                         '{:.2f}'.format(eff_vec[x_idx]),
                         ha = 'center', fontweight='bold', fontsize = 'large',
                         Bbox = dict(facecolor = colordict[idx_s], alpha =.4))
    
    #plt.xlabel('#GPUs', fontweight='bold', fontsize = 12)
    plt.ylabel(get_ylabel(metric), fontsize = 14)
    plt.ylim([0, ymax * 1.17])
    plt.xticks(fontsize=14 , fontweight = 'bold')
    
    plt.yticks(fontsize=14)
    
    if metric == 'nsr_and_ntr':
        plt.xticks([r + barWidth for r in range(len(mean_for_current_metric_current_solver))], r1_ticks)
        plt.title('{} out of {} for {} classification with {}'.format(dict_metric_to_title[metric], 
                  ntr_list[0], dict_dataset_to_title[dataset], dict_net_to_title[net_type]),
                  fontsize = 15.5)
        plt.legend(loc = 'upper right', ncol = 6 , fontsize = 10.5)
    else:
        plt.xticks([r + barWidth for r in range(len(mean_for_current_metric_current_solver))], r1_ticks)
        plt.title('{} for {} classification with {}'.format(dict_metric_to_title[metric],
                                                            dict_dataset_to_title[dataset], dict_net_to_title[net_type]),
                  fontsize = 15.5)
        plt.legend(loc = 'upper right', ncol = ncol_legend, fontsize = 10.5)
    plt.tight_layout()
imgname = savepath + '{}_{}_crit_{}_bs_{}.pkl'.format(dataset, net_type, t_acc_criterion, batch_size)
plt.savefig(imgname + '.eps', format = 'eps')
plt.savefig(imgname + '.png', format = 'png')

plt.show()
############ iterate over dict object to create barplot #######################