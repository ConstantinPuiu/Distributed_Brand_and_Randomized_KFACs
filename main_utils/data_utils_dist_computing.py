import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    elif dataset == 'imagenette':
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
                         transforms.transforms.Resize([128,128]), #transforms.Resize([224,224]), #Resize([32,32])
                         transforms.RandomHorizontalFlip(), 
                         transforms.RandomCrop(128, padding=4, padding_mode='reflect'), #transforms.RandomCrop(224, padding=4, padding_mode='reflect'), #RandomCrop(32, padding=4, padding_mode='reflect')
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
       
        transform_test = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor(), transforms.Normalize(*stats)]) #transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize(*stats)])
        
    elif dataset == 'imagenet':
        stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])
        
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])

        
        
    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, collation_fct = None, root='./data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    # adapt root folder based on chosen dataset
    root = root + '/' + dataset + '_data/'
    # get dataset
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'imagenet':
        trainset = torchvision.datasets.ImageNet(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.ImageNet(root=root, train=False, download=True, transform=transform_test)
        # link to download imagenet from to work with this function : https://image-net.org/challenges/LSVRC/2012/2012-downloads.php . 
        # Link to download webpage (which requeires approval) also at https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html.
        # see Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein,
        # Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014
    
    """if dataset == 'imagenette':
        # data_dir = '/content/gdrive/My Drive/datasets/imagenette/imagenette2' (root should be this)
        
        trainset = ImageFolder(root + '/train', transform = transform_train) #torchvision.datasets.ImageNet(root + '/train', transform = transform_train) # 
        testset = ImageFolder(root + '/val', transform = transform_test) # torchvision.datasets.ImageNet(root + '/val', transform = transform_test) # 
        
        #train_features = torch.load(featpath)
        #train_labels   =  torch.load(labpath)
        #trainset = torch.utils.data.TensorDataset(train_features, train_labels)"""

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    
    num_classes = len(trainset.classes)
    return trainset, testset, num_classes
