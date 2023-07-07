from torch.optim import SGD

class SGD_wrapped_like_KFAC:
    def __init__(self, model_parameters, lr,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum = 0.0, dampening = 0.0, 
                 weight_decay = 0.0, nesterov = False):
        
        self.steps = 0 # used to count number of staps taken
        self.epoch_number = 1
        self.lr_function = lr_function # save handlt to get lr schedule
        self.lr = lr_function(epoch_n = self.epoch_number, iter_n = self.steps )
        
        #### dummy attributes
        self.acc_stats = False # dummy variable that will be set on-off while doing nothing s.t. we can conveniently use the FAC train engine
        self.TCov = 10000000  #dummy TCov period s.t. we can use the same engine as KFAC
        
        # super init
        self.internal_SGD_optimizer = SGD(model_parameters, lr = lr,
                                     momentum = momentum, dampening = dampening, 
                                     weight_decay = weight_decay, nesterov = nesterov)
        
    def step(self, epoch_number, error_savepath, closure = None):
        
        #### set epoch number in self, and choose lr according to schedule
        self.steps += 1
        self.epoch_number = epoch_number
        self.lr = self.lr_function(epoch_n = self.epoch_number, iter_n = self.steps )
        for g in self.internal_SGD_optimizer.param_groups:
            g['lr'] = self.lr
        
        # take actual SGD step
        self.internal_SGD_optimizer.step(closure)
    
    def zero_grad(self, set_to_none):
        self.internal_SGD_optimizer.zero_grad(set_to_none)
        