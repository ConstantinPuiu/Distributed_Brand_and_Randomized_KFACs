import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

from Distributed_Brand_and_Randomized_KFACs.main_utils.scalable_vgg_models import vgg11_bn_scalable, vgg13_bn_scalable, vgg16_bn_scalable
from Distributed_Brand_and_Randomized_KFACs.main_utils.vgg_model import vgg16_bn_less_maxpool, vgg16_bn
from Distributed_Brand_and_Randomized_KFACs.main_utils.FC_nets import FC_net_for_CIFAR10
  
def get_network(network, dropout = False, **kwargs):
    '''networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn

    }'''   
    networks = {'vgg16_bn' : vgg16_bn,
                'vgg16_bn_less_maxpool': vgg16_bn_less_maxpool,
                'FC_net_for_CIFAR10': FC_net_for_CIFAR10,
                'vgg11_bn_scalable': vgg11_bn_scalable, 
                'vgg13_bn_scalable': vgg13_bn_scalable ,
                'vgg16_bn_scalable': vgg16_bn_scalable
                }
    
    # select network
    Net = networks[network](**kwargs)
    
    if dropout == True :
        if network == 'vgg16_bn_less_maxpool':
            Net.classifier = nn.Sequential(
                                                  nn.Linear(512 * 8 * 4, 512 * 4),
                                                  nn.ReLU(inplace = False),
                                                  nn.Dropout(), # p =0.5  by default
                                                  nn.Linear(512 * 4, kwargs.get('num_classes', 10) ),
                                              )
        elif network == 'vgg16_bn':
            Net.classifier = nn.Sequential(
                                                  nn.Linear(512 * 1 * 1, 512 * 1),
                                                  nn.ReLU(inplace = False),
                                                  nn.Dropout(), # p =0.5  by default
                                                  nn.Linear(512 * 1, kwargs.get('num_classes', 10) ),
                                              )
        elif network == 'FC_net_for_CIFAR10':
            Net.classifier = nn.Sequential(
                                                  nn.Linear(512 * 8 * 4, 512 * 4),
                                                  nn.ReLU(inplace = False),
                                                  nn.Dropout(), # p =0.5  by default
                                                  nn.Linear(512 * 4, kwargs.get('num_classes', 10) ),
                                              )
        
        elif network in ['vgg11_bn_scalable', 'vgg13_bn_scalable', 'vgg16_bn_scalable']:
            Net.classifier = nn.Sequential(
                                                  nn.Linear(512 * 1 * kwargs.get('Network_scalefactor', 1), 512 * 1),
                                                  nn.ReLU(inplace = False),
                                                  nn.Dropout(), # p =0.5  by default
                                                  nn.Linear(512 * 1, kwargs.get('num_classes', 10) ),
                                              )
        else:
            raise NotImplementedError('Dropout only implemented for [vgg16_bn_less_maxpool] and [FC_net_for_CIFAR10] for now!')
    
    return Net
