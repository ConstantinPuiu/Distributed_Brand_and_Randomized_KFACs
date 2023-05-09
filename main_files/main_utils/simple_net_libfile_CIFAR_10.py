import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Distributed_Brand_and_Randomized_KFACs.main_utils.vgg_model import vgg16_bn_less_maxpool

  
def get_network(network, dropout = False, **kwargs):
    '''networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn

    }'''   
    networks = {'vgg16_bn_less_maxpool': vgg16_bn_less_maxpool}
    
    # select network
    Net = networks[network](**kwargs)
    
    if dropout == True :
        if network == 'vgg16_bn_less_maxpool':
            Net.classifier = nn.Sequential(
                                                  nn.Linear(512 * 8 * 4, 512 * 4),
                                                  nn.ReLU(True),
                                                  nn.Dropout(), # p =0.5  by default
                                                  nn.Linear(512 * 4, 10),
                                              )
        else:
            raise NotImplementedError('Dropout only implemented for [vgg16_bn_less_maxpool] for now!')
    
    return Net
