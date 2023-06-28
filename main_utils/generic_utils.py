import torch
import torch.distributed as dist
import time

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
        self.accum_val += new_val
        self.n_pts += new_n_pts
    
    def avg(self):
        dist.all_reduce(self.accum_val, async_op=False)
        dist.all_reduce(self.n_pts, async_op=False)
        return self.accum_val.cpu() / self.n_pts.cpu()
############ END: helper class for test funciton ##########################

def test(test_loader, model, loss_fn, rank, world_size, epoch):
    # rank here is the GPU rank (idx of GPU, NOT rsvd rank or anything like that)
    model.eval()
    test_loss = Metric('test_loss', rank)
    test_acc = Metric('test_acc', rank)
    
    with torch.no_grad():
        for idx, (x, y_target) in enumerate(test_loader): # each GPU has it's own test loader
            y_pred = model(x)
            current_bsz = y_target.shape[0] # making sure leftover batches (i.e. a batch of 190 when bs = 256 is accounted for correctly)
            
            #### loss sum-over_pts-in-batch computation
            loss_increment = loss_fn(y_pred, y_target) * current_bsz #loss fct returns average over batch and we want sum, hence the multiplication with current_bsz
            
            #number of correct pts computation (for test-acc)
            pred = y_pred.data.max(1, keepdim=True)[1]
            acc_increment = pred.eq(y_target.data.view_as(pred)).sum() # acc increment is just the number of correctly classified pts
            
            # update metrics
            test_loss.update(new_val = loss_increment, new_n_pts = current_bsz)
            test_acc.update(new_val = acc_increment, new_n_pts = current_bsz)
    tl = test_loss.avg()    
    ta = test_acc.avg(); ta = 100 * ta # display as percentage
    #Test is currently printing. TO DO: store to list and save for future plots
    print('Rank {} / ws = {}. Epoch = {} :  test_loss = {:.5f}, test_accuracy = {:.4f}%\n'.format(rank, world_size, epoch + 1, tl, ta))
    #return tl, ta
    
def train_n_epochs(model, optimizer, loss_fn, train_set, test_set, schedule_function, args, len_train_set, rank, world_size):
    # args contains how many epchs to train
    total_time = 0
    ########################## TRAINING LOOP: over epochs ######################################################
    for epoch in range(0, args.n_epochs):
        tstart = time.time()
        # if we are using DistributedSampler, we have to tell it which epoch this is
        #dataloader.sampler.set_epoch(epoch)
        
        ######### setting parameters according to SCHEDULES ###################
        schedule_function(optimizer, epoch)
        ######### END: setting parameters according to SCHEDULES ##############
        
        ################### TRAINING LOOP: over batches #######################
        for jdx, (x,y) in enumerate(train_set):#dataloader):
            #print('\ntype(x) = {}, x = {}, x.get_device() = {}\n'.format(type(x), x, x.get_device()))
            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            #label = x['label']
            if optimizer.steps % optimizer.TCov == 0: #KFAC_matrix_update_frequency == 0:
                optimizer.acc_stats = True
            else:
                optimizer.acc_stats = False
                
            loss = loss_fn(pred, y)#label)
            
            #print('rank = {}, epoch = {} at step = {} ({} steps per epoch) has loss.item() = {}'.format(rank, jdx, optimizer.steps, len_train_set, loss.item()))
                
            loss.backward()
                #with open('/data/math-opt-ml/chri5570/initial_trials/2GPUs_test_output.txt', 'a+') as f:
                #    f.write('Rank (GPU number) {} at batch {}:'.format(rank, jdx) + str(dist.get_rank())+ ', epoch ' +str(epoch+1) + ', loss: {}\n'.format(str(loss.item())))
            optimizer.step(epoch_number = epoch + 1, error_savepath = None)
        ################ END:  TRAINING LOOP: over batches ####################
        
        tend = time.time(); total_time += (tend- tstart)
        
        ############ test every few epochs during training ####################
        if ((epoch + 1) % args.test_every_X_epochs) == 0:   
            if ( epoch + 1 < args.n_epochs ) or ( args.test_at_end == False): # if we were gonna test at the end and this is the final epoch, don't test here to avoid duplicates
                print('Rank = {}. Testing at epoch = {}... \n'.format(rank, epoch + 1))
                test(test_loader = test_set, model = model, loss_fn = loss_fn, rank = rank, world_size = world_size, epoch = epoch) 
                model.train() # put model back into training mode
        ############ END: test every few epochs during training ###############
    
    ##################### END : TRAINING LOOP: over epochs ####################################################
    
    print('TIME: {:.3f} s. Rank (GPU number) {} at batch {}, total steps optimizer.steps = {}:'.format(total_time, rank, jdx, optimizer.steps) + ', epoch ' +str(epoch + 1) + ', instant train-loss: {:.5f}\n'.format(loss.item()))

    ####### test at the end of training #####
    if args.test_at_end == True: 
        print('Rank = {}. Testing at the end (i.e. epoch = {})... \n'.format(rank, args.n_epochs + 1))
        test(test_loader = test_set, model = model, loss_fn = loss_fn, rank = rank, world_size = world_size, epoch = args.n_epochs - 1) 
    ## END:  test at the end of training ####    