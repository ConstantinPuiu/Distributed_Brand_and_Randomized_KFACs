def get_l_rate_function(lr_schedule_type, base_lr, lr_decay_rate, lr_decay_period, 
                        auto_scale_forGPUs_and_BS, n_GPUs, batch_size, dataset = None):
    # returns function handle to be called with just epoch parameter
    # exp, cos, and staircase inpired from SENG code : https://github.com/yangorwell/SENG/blob/main/Pytorch/cifar10/main_seng.py
    if auto_scale_forGPUs_and_BS == True:
        total_batch_size_div_256 = n_GPUs * batch_size  / 256
        lr_sqrt_scale_factor = (total_batch_size_div_256 ** 0.5) 
    else:
        total_batch_size_div_256 = lr_sqrt_scale_factor = 1
    
    if lr_schedule_type == 'constant':
        def lr_sch_constant(epoch_n, iter_n = None):
            lr = base_lr
            return lr * lr_sqrt_scale_factor
        return lr_sch_constant
    
    elif  lr_schedule_type == 'staircase':
        def lr_sch_stair(epoch_n, iter_n = None):
            epoch_n = epoch_n / total_batch_size_div_256
            lr = base_lr * (lr_decay_rate**(epoch_n // lr_decay_period))
            return lr * lr_sqrt_scale_factor
        return lr_sch_stair
    
    elif lr_schedule_type == 'cos':
        import math
        def lr_sch_cos(epoch_n, iter_n = None):
            epoch_n = epoch_n / total_batch_size_div_256
            if epoch_n < lr_decay_period:
                lr = 0.001 + 0.5 * (base_lr - 0.001) * (1 + math.cos(epoch_n / lr_decay_period * math.pi))
            else:
                lr = 0.0005
            return lr * lr_sqrt_scale_factor
        return lr_sch_cos
    
    elif lr_schedule_type == 'exp':
        def lr_sch_exp(epoch_n, iter_n = None):
            epoch_n = epoch_n / total_batch_size_div_256
            lr = base_lr * (1.0 - epoch_n/lr_decay_period)**lr_decay_rate
            return lr * lr_sqrt_scale_factor
        return lr_sch_exp
    
    elif lr_schedule_type == 'from_file':
        return get_l_rate_function_for_dataset(dataset, total_batch_size_div_256, lr_sqrt_scale_factor)
    
    else:
        print('\n\n\ !!! WARNING !!!! : lr_schedule_type = {} does NOT have an implemented lr schedule. Defaulting to CIFAR10 lr schedule in file lrfct.py!\n\n\n'.format(dataset))
        return get_l_rate_function_for_dataset('cifar10', total_batch_size_div_256, lr_sqrt_scale_factor)


############## Below : from-file per dataset lr manual schedules #####################
        
def get_l_rate_function_for_dataset(dataset, total_batch_size_div_256, lr_sqrt_scale_factor): # returns function handle to correct lr_fct depending on dataset
    
    if dataset not in ['MNIST', 'SVHN', 'cifar10', 'cifar100', 'imagenet', 'imagenette_fs_v2']:
        # cifar10 lr funciton is set to default if new datasets are encountered: sometwhat arbitrary. Change if needed
        print('\n\n\ !!! WARNING !!!! : dataset = {} does NOT have an implemented lr schedule. Defaulting to CIFAR10 lr schedule!\n\n\n'.format(dataset))
        dataset = 'cifar10'
        
    if dataset == 'MNIST':
        def l_rate_function_MNIST(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor
        
        return l_rate_function_MNIST 
    
    elif dataset == 'SVHN':
        def l_rate_function_SVHN(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor
        return l_rate_function_SVHN
    
    elif dataset == 'cifar10':
        def l_rate_function_cifar10(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor
        
        return l_rate_function_cifar10
    
    elif dataset == 'cifar100': 
        def l_rate_function_cifar100(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor
   
        return l_rate_function_cifar100
    
    elif dataset == 'imagenet':      
        def l_rate_function_imagenet(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor
   
        return l_rate_function_imagenet
    
    elif dataset == 'imagenette_fs_v2':
        def l_rate_function_imagenette_fs_v2(epoch_n, iter_n):
            #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
            epoch_n = epoch_n // total_batch_size_div_256
            if epoch_n <= 1:
                if iter_n < 3:
                    lr = 0.3
                else:
                    lr = 0.3
            elif epoch_n == 2:
                lr = 0.2
            elif epoch_n >= 3 and epoch_n < 7:
                lr = 0.1
            elif epoch_n >= 7 and epoch_n < 13:
                lr = 0.1
            elif epoch_n >= 13 and epoch_n < 18:
                lr = 0.03
            elif epoch_n >= 18 and epoch_n < 27:
                lr = 0.01
            elif epoch_n >= 27 and epoch_n < 40:
                lr = 0.003
            elif epoch_n >= 40:
                lr = 0.001
            return lr * lr_sqrt_scale_factor

        return l_rate_function_imagenette_fs_v2



