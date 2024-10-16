'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11_scalable', 'vgg11_bn_scalable', 'vgg13_scalable', 'vgg13_bn_scalable', 
    'vgg16_scalable', 'vgg16_bn_scalable',
    'vgg19_bn_scalable', 'vgg19_scalable',
]


"""model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}
"""

class VGG(nn.Module):

    def __init__(self, features, Network_scalefactor = 1, num_classes=1000, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(int(512 * Network_scalefactor), num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, Network_scalefactor = 1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M1Dh':
            layers += [nn.MaxPool2d(kernel_size= (1,2), stride = (1,2))]
        elif v == 'M1Dv':
            layers += [nn.MaxPool2d(kernel_size= (2,1), stride = (2,1))]
        elif v == 'M1samesize': # dummy layerfor now, does nothing - keep to change easy to 'D' cfg!
            #layers += [nn.MaxPool2d(kernel_size= (2,2), stride = (1,1))]
            pass
        else:
            v = int(v * Network_scalefactor)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = False)]
            else:
                layers += [conv2d, nn.ReLU(inplace = False)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_less_maxpooled' : [64, 64, 'M1Dh', 128, 128, 'M1Dv', 256, 256, 256, 'M1Dh', 512, 512, 512, 'M1Dv', 512, 512, 512, 'M1Dh'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg11_bn_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True, Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg13_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor **kwargs)
    return model


def vgg13_bn_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True, Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg16_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg16_bn_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model
    
def vgg16_bn_less_maxpool_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D_less_maxpooled'], batch_norm=True, Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg19_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model


def vgg19_bn_scalable(Network_scalefactor = 1, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True, Network_scalefactor = Network_scalefactor), Network_scalefactor = Network_scalefactor, **kwargs)
    return model
