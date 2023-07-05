def get_l_rate_function_for_dataset(dataset): # returns function handle to correct lr_fct depending on dataset
    if dataset == 'MNIST':
        return l_rate_function_MNIST 
    elif dataset == 'SVHN':
        return l_rate_function_SVHN
    elif dataset == 'cifar10':
        return l_rate_function_cifar10
    elif dataset == 'cifar100':
        return l_rate_function_cifar100
    elif dataset == 'imagenet':
        return l_rate_function_imagenet
    elif dataset == 'imagenette_fs_v2':
        return l_rate_function_imagenette_fs_v2
    else: # cifar10 lr funciton is set to default if new datasets are encountered: sometwhat arbitrary. Change if needed
        print('\n\n\ !!! WARNING !!!! : dataset = {} does NOT have an implemented lr schedule. Defaulting to CIFAR10 lr schedule!\n\n\n'.format(dataset))
        l_rate_function_cifar10


def l_rate_function_MNIST(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 

def l_rate_function_SVHN(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 


def l_rate_function_cifar10(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 
    
def l_rate_function_cifar100(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 
    
def l_rate_function_imagenet(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 
    
def l_rate_function_imagenette_fs_v2(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    total_batch_size_div_256 = n_GPUs * batch_size  / 256
    epoch_n = epoch_n * total_batch_size_div_256
    if epoch_n == 1:
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
    
    return lr * (total_batch_size_div_256 ** 0.5) 



