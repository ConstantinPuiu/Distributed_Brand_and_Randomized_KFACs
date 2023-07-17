import torch
import torch.distributed as dist
import time
from tqdm import tqdm

from Distributed_Brand_and_Randomized_KFACs.main_utils.simple_net_libfile_CIFAR_10 import get_network
import torchvision.models as torchVmodels
import Distributed_Brand_and_Randomized_KFACs.main_utils.resnet_for_CIFAR10 as resnet_for_CIFAR10
from Distributed_Brand_and_Randomized_KFACs.main_utils.simple_net_libfile import Net as simple_MNIST_net

# instantiate the model(it's your own model) and move it to the right device
def get_net_main_util_fct(net_type, rank, num_classes = 10):
    if '_corrected' in net_type:
        print('Using corrected resnet is only for CIFAR10, and your num_classes was {} != 10. Please use (standard) resne with this dataset'.format(num_classes))
        # strictly speaking, could make resnet_corrected work witha ny num_classes by setting model.linear to an C with the desired number of classes. We don't do that as we can just use standard resnets
    if net_type == 'Simple_net_for_MNIST':
        model = simple_MNIST_net().to(rank)
    elif net_type == 'VGG16_bn_lmxp':
        model = get_network('vgg16_bn_less_maxpool', dropout = True, #depth = 19,
                     num_classes = num_classes,
                     #growthRate = 12,
                     #compressionRate = 2,
                     widen_factor = 1).to(rank)
    elif net_type == 'FC_CIFAR10':
        model = get_network('FC_net_for_CIFAR10', dropout = True, #depth = 19,
                     num_classes = num_classes).to(rank)
    elif net_type == 'resnet18':
        model = torchVmodels.resnet18( num_classes = num_classes ).to(rank)
    elif net_type == 'resnet50':
        model = torchVmodels.resnet50( num_classes = num_classes ).to(rank)
    elif net_type == 'resnet101':
        model = torchVmodels.resnet101( num_classes = num_classes ).to(rank)
    elif net_type == 'resnet20_corrected':
        model = resnet_for_CIFAR10.resnet20().to(rank)
    elif net_type == 'resnet32_corrected':
        model = resnet_for_CIFAR10.resnet32().to(rank)
    elif net_type == 'resnet44_corrected':
        model = resnet_for_CIFAR10.resnet44().to(rank)
    elif net_type == 'resnet56_corrected':
        model = resnet_for_CIFAR10.resnet56().to(rank)        
    else:
        raise ValueError('Net of type: net_type = {} Not implemented'.format(net_type) )
    
    return model

############ helper class for test funciton ##############################
class Metric:
    def __init__(self, name, rank):
        # a metric is kept for EACH GPU, then averaged over GPUs at the very end!
        self.name = name
        self.accum_val = torch.tensor(0.0, device = torch.device('cuda:{}'.format(rank)))
        self.n_pts = torch.tensor(0.0, device = torch.device('cuda:{}'.format(rank)))
        # n_pts carries how many datapoints has each GPU seen. 
        # All-reducing at the end gives the total number of points over all GPUs, which is the correct denominator for averaging
        # note taht no worldsize is required as this is implicitly included in the allreducing of n_pts
    
    def update(self, new_val, new_n_pts):
        with torch.no_grad():
            self.accum_val += new_val
            self.n_pts += new_n_pts
    
    def avg(self):
        dist.all_reduce(self.accum_val, async_op=False)
        dist.all_reduce(self.n_pts, async_op=False)
        return self.accum_val.cpu() / self.n_pts.cpu()
    
    def zero_out(self): # zero out - used for trianing metrics only to restart them rather than re-initialize at each epoch
        self.accum_val.zero_()
        self.n_pts.zero_()
############ END: helper class for test funciton ##########################

###############################################################################
################## helpers for easily running metric objects ##################
###############################################################################
def update_acc_and_loss_obj( loss_metric_obj, acc_metric_obj, loss_fn, y_pred, y_target, loss_increment_raw = None):
    current_bsz = y_target.shape[0] # making sure leftover batches (i.e. a batch of 190 when bs = 256 is accounted for correctly)
    
    #### loss sum-over_pts-in-batch computation
    if loss_increment_raw == None: # allowing the possibility to bypass having to call the loss fct again in case we called it outside this function
        #loss_increment_raw is meant to be exactly loss_fn(y_pred, y_target) called outside
        loss_increment = loss_fn(y_pred, y_target) * current_bsz #loss fct returns average over batch and we want sum, hence the multiplication with current_bsz
    else:
        loss_increment = loss_increment_raw * current_bsz
    
    #number of correct pts computation (for test-acc)
    acc_increment = compute_n_correct_predictions(y_pred, y_target) # acc increment is just the number of correctly classified pts
    
    # update metrics
    loss_metric_obj.update(new_val = loss_increment, new_n_pts = current_bsz)
    acc_metric_obj.update(new_val = acc_increment, new_n_pts = current_bsz)

############ helper function for computing accuracy ###########################
def compute_n_correct_predictions(y_pred, y_target):
    pred = y_pred.data.max(1, keepdim=True)[1]
    return pred.eq(y_target.data.view_as(pred)).sum()
############ END : helper function for computing accuracy #####################

###############################################################################
############ END: helpers for easily running metric objects ###################
###############################################################################
    
def test(test_loader, model, loss_fn, args, stored_metrics_object, rank, world_size, epoch, time_to_epoch_end_test, test_loss_obj, test_acc_obj):
    # rank here is the GPU rank (idx of GPU, NOT rsvd rank or anything like that)
    model.eval()
    
    len_test_loader = len(test_loader)
    with tqdm(
        total = len_test_loader,
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='\nTesting at epoch {} [in {} iterations].'.format(epoch + 1, len_test_loader),
        disable = (not args.print_tqdm_progress_bar) or (rank != 1), # only rank1 will be verbose
        miniters = int(len_test_loader / 2)
    ) as t:
        with torch.no_grad():
            for idx, (x, y_target) in enumerate(test_loader): # each GPU has it's own test loader
                x, y_target = x.to(torch.device('cuda:{}'.format(rank))), y_target.to(torch.device('cuda:{}'.format(rank)))
                y_pred = model(x)
                
                ##### update loss and acc metric objects ##########
                update_acc_and_loss_obj( loss_metric_obj = test_loss_obj, acc_metric_obj = test_acc_obj,
                                        loss_fn = loss_fn, y_pred = y_pred, y_target = y_target)
                
                # update tqdm progress bar
                t.update(1)
    tl = test_loss_obj.avg()    
    ta = test_acc_obj.avg(); ta = 100 * ta # display as percentage
    
    #Test is currently printing. TO DO: store to list and save for future plots
    print('\nRank {} / ws = {}. Epoch = {} :  test_loss = {:.5f}, test_accuracy = {:.4f}%\n'.format(rank, world_size, epoch + 1, tl, ta))
    
    ##### update and return stored metrics object if required. If not None is returned
    if args.store_and_save_metrics == True and rank == 0: # do only storing on GPU #0 process (still on CPU to avoid duplicates stored)
        """ IMPORTANT NOTE: ONLY THE TIME MEASURED by GPU0 is considered
        But the all-reduced test-acc and test-loss is considered (all reduce hidden in .avg() method"""
        stored_metrics_object.update_lists({'epoch_number_test': epoch + 1, 'test_loss': tl , 'test_acc': ta , 'time_to_epoch_end_test': time_to_epoch_end_test})
    else: # else want to return stored_metrics_object = None, but stored_metrics_object is already None in this case, so passing
        pass 
    ########### zero out test metric objects to have them fresh at the beginning of bext test
    test_loss_obj.zero_out()
    test_acc_obj.zero_out()    
    
    #return tl, ta
    return stored_metrics_object, tl, ta

################ helper class for storing metrics  (all in 1 object ########### =========================================
class stored_metrics:
    def __init__(self, metrics_list):
        # metrics_list is a list of strings
        self.metrics_dict = {} # possible keys: 'test_loss', 'test_acc', 'time_to_epoch'
        for metr in metrics_list:
            self.metrics_dict[metr] = []
    
    def update_lists(self, metrics_to_update_dict):# run as  for eg stored_metrics_bject.update_lists({'test_loss': 0.02 , 'test_acc': 99.4 , 'time_to_epoch': 1})
        #metrics_to_update_dict is of form key (metric, i.e. test acc): value
        for metr in metrics_to_update_dict.keys():
            self.metrics_dict[metr].append(metrics_to_update_dict[metr])
        
    def get_device_names_and_store_in_object(self, world_size):
        self.metrics_dict['GPU_names'] = []
        for idx in range(0, world_size):
            self.metrics_dict['GPU_names'].append(torch.cuda.get_device_name(idx))            
    
    def save_metrics(self, metrics_save_path, dataset, net_type, solver_name, nGPUs, batch_size, run_seed):
        # get date ###########################################
        import os
        from datetime import datetime
        date_time_now = datetime.now()
        date_now = str(date_time_now.date())
        ######################################################
        # make a dir with today's date if not already exists ################################
        metrics_save_path = metrics_save_path + '/' + date_now
        if os.path.exists(metrics_save_path):
            pass # if the path already exists then do nothing
        else: # if it doesnt exist, create it
            os.mkdir(metrics_save_path)
        ######################################################################################
        for metr in self.metrics_dict:
            torch.save(obj = self.metrics_dict[metr], f = metrics_save_path + '/{}_{}_{}_nGPUs_{}_Bsize_{}_{}_{}.pt'.format( dataset, net_type, solver_name, nGPUs, batch_size, metr, run_seed) )
    
    def print_metrics(self):
        for metr in self.metrics_dict:
            print('Metric {}, data: {}\n'.format(metr, self.metrics_dict[metr]))
################ END: helper class for storing metrics ######################## =========================================

def train_n_epochs(model, optimizer, loss_fn,  train_loader, train_sampler, test_loader,  schedule_function,
                   args, len_train_loader, rank, world_size, optim_type = 'KFAC'):
    # optim_type is only used to know whether we have to (and do) set optimizer.acc_stats = True and false
    # function returns stored_metrics_object # stored_metrics_object is None if we do not store metrics
    # args contains how many epchs to train
    
    ####### initialize object to store metrics if required ################
    if args.store_and_save_metrics == True and rank == 0:
        # initialize stored_metrics_object
        stored_metrics_object = stored_metrics(['epoch_number_train' , 'train_loss' , 'train_acc' , 'time_to_epoch_end_train',
                                                'epoch_number_test', 'test_loss' , 'test_acc' , 'time_to_epoch_end_test'])
    else:
        stored_metrics_object = None
    ####### initialize object to store metrics if required ################
         
    total_time = 0
    
    ########### iniitalize Metric objects #####################################
    train_loss_obj = Metric('train_loss', rank)
    train_acc_obj = Metric('train_acc', rank)
    test_loss_obj = Metric('test_loss', rank)
    test_acc_obj = Metric('test_acc', rank)
    ########### END: iniitalize Metric objects ################################

    ########################## TRAINING LOOP: over epochs ######################################################
    with tqdm(
        total = args.n_epochs,
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='\nTraining a total of {} epochs.'.format(args.n_epochs),
        disable =  (not args.print_tqdm_progress_bar) or (rank != 1), # only rank1 will be verbose
    ) as t:
        for epoch in range(0, args.n_epochs):
            tstart = time.time()
            train_sampler.set_epoch(epoch)
            # if we are using DistributedSampler, we have to tell it which epoch this is
            #dataloader.sampler.set_epoch(epoch)
            
            ######### setting parameters according to SCHEDULES ###################
            schedule_function(optimizer, epoch)
            ######### END: setting parameters according to SCHEDULES ##############
            
            ################### TRAINING LOOP: over batches #######################
            for jdx, (x,y) in enumerate(train_loader):#dataloader):
                ###send data to GPU
                x, y = x.to(torch.device('cuda:{}'.format(rank))), y.to(torch.device('cuda:{}'.format(rank)))
                #print('\ntype(x) = {}, x = {}, x.get_device() = {}\n'.format(type(x), x, x.get_device()))
                optimizer.zero_grad(set_to_none=True)
    
                pred = model(x)
                #label = x['label']
                if optimizer.steps % optimizer.TCov == 0: #KFAC_matrix_update_frequency == 0:
                    optimizer.acc_stats = True
                else:
                    optimizer.acc_stats = False
                    
                loss = loss_fn(pred, y)#label)
                
                ##### update loss and acc metric objects ##########
                update_acc_and_loss_obj( loss_metric_obj = train_loss_obj, acc_metric_obj = train_acc_obj,
                                        loss_fn = loss_fn, y_pred = pred, y_target = y, 
                                        loss_increment_raw = loss)
                #print('rank = {}, epoch = {} at step = {} ({} steps per epoch) has loss.item() = {}'.format(rank, epoch, optimizer.steps, len_train_loader, loss.item()))
                    
                loss.backward()
                    #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
                    #    f.write('Rank (GPU number) {} at batch {}:'.format(rank, jdx) + str(dist.get_rank())+ ', epoch ' +str(epoch+1) + ', loss: {}\n'.format(str(loss.item())))
                
                optimizer.step(epoch_number = epoch + 1, error_savepath = None)
            ################ END:  TRAINING LOOP: over batches ####################
            
            #### end epoch-time measurement
            tend = time.time(); total_time += (tend- tstart)
            
            ############# all-reduce, store, and save training metric objects ################
            train_l = train_loss_obj.avg()    
            train_a = train_acc_obj.avg(); train_a = 100 * train_a # display as percentage
            #Test is currently printing. TO DO: store to list and save for future plots
            print('\nRank {} / ws = {}. Epoch = {} :  train_loss = {:.5f}, train_accuracy = {:.4f}%\n'.format(rank, world_size, epoch + 1, train_l, train_a))
            
            ##### update and return stored metrics object if required. If not None is returned
            if args.store_and_save_metrics == True and rank == 0: # do only storing on GPU #0 process (still on CPU to avoid duplicates stored)
                """ IMPORTANT NOTE: ONLY THE TIME MEASURED by GPU0 is considered
                But the all-reduced train-acc and train-loss is considered (all reduce hidden in .avg() method
                Further note that the train metrics are the averga over the entire epoch, so might be slightly stale as compared to en-dof-epoch results"""
                stored_metrics_object.update_lists({'epoch_number_train': epoch + 1, 'train_loss': train_l , 'train_acc': train_a ,
                                                    'time_to_epoch_end_train': total_time})
            else: # else want to return stored_metrics_object = None, but stored_metrics_object is already None in this case, so passing
                pass 
            ########### zero out test metric objects to have them fresh at the beginning of bext test
            train_loss_obj.zero_out()
            train_acc_obj.zero_out()    
            ############# END: all-reduce, store, and save training metric objects ###########
            
            #####update tqdm
            t.update(1)
            
            ############ test every few epochs during training ####################
            if ((epoch + 1) % args.test_every_X_epochs) == 0:   
                if ( epoch + 1 < args.n_epochs ) or ( args.test_at_end == False): # if we were gonna test at the end and this is the final epoch, don't test here to avoid duplicates
                    print('\nRank = {}. Testing at epoch = {}... \n'.format(rank, epoch + 1))
                    stored_metrics_object, _, ta = test(test_loader = test_loader, model = model, loss_fn = loss_fn, args = args, 
                                                                stored_metrics_object = stored_metrics_object, 
                                                                rank = rank, world_size = world_size, epoch = epoch,
                                                                time_to_epoch_end_test = total_time,
                                                                test_loss_obj = test_loss_obj, test_acc_obj = test_acc_obj) 
                    
                    if args.stop_at_test_acc == True and ta >= args.stopping_test_acc: # test-loss based stopping criterion to be implemented
                        print('Rank {}, n_GPUs = {}: Training stopped with\
                              test-acc criterion satisfied ( since args.stop_at_test_acc = {}, \
                              args.stopping_test_acc = {} and attained test accuracy is {}\
                                  ) !'.format(rank, world_size, args.stop_at_test_acc, 
                                              args.stopping_test_acc, ta))
                        break
                    model.train() # put model back into training mode
            ############ END: test every few epochs during training ###############
    
    ##################### END : TRAINING LOOP: over epochs ####################################################
    print('\n!!!\nTIME: {:.3f} s. Rank (GPU number) {} at batch {}, total steps optimizer.steps = {}:'.format(total_time, rank, jdx, optimizer.steps) + ', epoch ' +str(epoch + 1) + ', instant train-loss: {:.5f}\n!!!\n'.format(loss.item()))
    ####### test at the end of training #####
    if args.test_at_end == True: 
        print('\nRank = {}. Testing at the end (i.e. epoch = {})... \n'.format(rank, epoch + 1))
        stored_metrics_object, _, _ = test(test_loader = test_loader, model = model, loss_fn = loss_fn, args = args, 
                                        stored_metrics_object = stored_metrics_object,
                                        rank = rank, world_size = world_size, epoch = args.n_epochs - 1,
                                        time_to_epoch_end_test = total_time, 
                                        test_loss_obj = test_loss_obj, test_acc_obj = test_acc_obj) 
    ## END:  test at the end of training ####    
    
    return stored_metrics_object