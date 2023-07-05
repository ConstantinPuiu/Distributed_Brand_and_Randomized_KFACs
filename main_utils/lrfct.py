def l_rate_function(epoch_n, n_GPUs, batch_size, iter_n):
    #### scale the number of epochs by the number of GPUs and batch-size to have the same lr based on steps
    epoch_n = epoch_n * n_GPUs * batch_size / 256
    
    if epoch_n == 1:
        if iter_n < 3:
            return 0.3
        else:
            return 0.3
    elif epoch_n == 2:
        return 0.2
    elif epoch_n >= 3 and epoch_n < 7:
        return 0.1
    elif epoch_n >= 7 and epoch_n < 13:
        return 0.1
    elif epoch_n >= 13 and epoch_n < 18:
        return 0.03
    elif epoch_n >= 18 and epoch_n < 27:
        return 0.01
    elif epoch_n >= 27 and epoch_n < 40:
        return 0.003
    elif epoch_n >= 40:
        return 0.001
