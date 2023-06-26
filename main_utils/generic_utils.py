from Distributed_Brand_and_Randomized_KFACs.main_utils.simple_net_libfile_CIFAR_10 import get_network
import torchvision.models as torchVmodels
import Distributed_Brand_and_Randomized_KFACs.main_utils.resnet_for_CIFAR10 as resnet_for_CIFAR10

# instantiate the model(it's your own model) and move it to the right device
def get_net_main_util_fct(net_type, rank, num_classes = 10):
    if '_corrected' in net_type:
        print('Using corrected resnet is only for CIFAR10, and your num_classes was {} != 10. Please use (standard) resne with this dataset'.format(num_classes))
        # strictly speaking, could make resnet_corrected work witha ny num_classes by setting model.linear to an C with the desired number of classes. We don't do that as we can just use standard resnets
    
    if net_type == 'VGG16_bn_lmxp':
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
        