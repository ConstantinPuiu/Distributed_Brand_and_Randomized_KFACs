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

    if dataset == 'cifar100':
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

    if dataset == 'imagenette':
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
                         transforms.transforms.Resize([128,128]), #transforms.Resize([224,224]), #Resize([32,32])
                         transforms.RandomHorizontalFlip(), 
                         transforms.RandomCrop(128, padding=4, padding_mode='reflect'), #transforms.RandomCrop(224, padding=4, padding_mode='reflect'), #RandomCrop(32, padding=4, padding_mode='reflect')
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

        transform_test = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor(), transforms.Normalize(*stats)]) #transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize(*stats)])
        
    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, collation_fct = None, root='./data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'imagenette':
        # data_dir = '/content/gdrive/My Drive/datasets/imagenette/imagenette2' (root should be this)
        
        trainset = ImageFolder(root + '/train', transform = transform_train) #torchvision.datasets.ImageNet(root + '/train', transform = transform_train) # 
        testset = ImageFolder(root + '/val', transform = transform_test) # torchvision.datasets.ImageNet(root + '/val', transform = transform_test) # 
        
        #train_features = torch.load(featpath)
        #train_labels   =  torch.load(labpath)
        #trainset = torch.utils.data.TensorDataset(train_features, train_labels)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset

    return trainset, testset
