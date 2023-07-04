import torch.distributed as dist
import random as rng
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from torchvision.datasets import ImageFolder

##############################################################################
############ DataSampler and Dataloader getter function ######################
##############################################################################

def get_data_loaders_and_s(data_root_path, dataset, batch_size, seed = -1):
    size = dist.get_world_size()
    #bsz = 256 #int(128 / float(size))
    if dataset in ['MNIST', 'SVHN', 'cifar10', 'cifar100', 'imagenet', 'imagenette_fs_v2']:
        train_set, test_set, num_classes = get_dataloader(dataset = dataset, train_batch_size = batch_size,
                                          test_batch_size = batch_size,
                                          root = data_root_path)
    else:
        raise NotImplementedError('dataset = {} is not implemeted'.format(dataset))
   
    ########### set seed ###########
    if seed == -1: # if seed is -1 do not seed
        # however we stll need to seed the random rng as otherwise our partitions will be wrong
        rng.seed(seed)
    else:
        rng.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)  # set ALL CUDA operations to deterministic
        # ideally we use this one instead of the one below, but have issues taht some random algorithms for CUDA cannot be made deterministic
        torch.backends.cudnn.deterministic=True # set all CUDA-Conv operations to deterministic
        
    ########### END: set seed ######
       
    ######## set kwarrgs options for dataloader ###############################
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 8, 'persistent_workers': True}
    #### END: set kwarrgs options for dataloader ##############################
        
    ################ build Train Sampler and Loader #######################################
    train_sampler = DistributedSampler(train_set,
                                       num_replicas = size,
                                       rank=dist.get_rank(),
                                       )
    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              sampler=train_sampler,
                              **kwargs,
                              )
    ########### END: build Train Sampler and Loader ########################################
    
    ################ build test Sampler and Loader #######################################
    val_sampler = DistributedSampler(
        test_set,
        num_replicas = size,
        rank = dist.get_rank(),
    )
    val_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        sampler=val_sampler,
        **kwargs,
    )
    ################ END: build test Sampler and Loader #################################    

    return train_sampler, train_loader, val_sampler, val_loader, batch_size, num_classes

def cleanup():
    dist.destroy_process_group()

##############################################################################
############  END : DataSampler and Dataloader getter function ###############
##############################################################################

def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'MNIST':
        transform_train = transforms.Compose([  transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))  ])
        transform_test = transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))  ])
    elif dataset == 'SVHN':
        transform_train = transforms.Compose([  transforms.Pad(padding=2),
                                                transforms.RandomCrop(size=(32, 32)),
                                                transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
                                                transforms.ToTensor()
                                            ])
        transform_test = transforms.Compose([  transforms.ToTensor(),
                                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))  ])
    
    elif dataset == 'cifar10':
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

    elif dataset == 'imagenette_fs_v2':
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        """
        transform_train = transforms.Compose([
                         transforms.transforms.Resize([128,128]), #transforms.Resize([224,224]), #Resize([32,32])
                         transforms.RandomHorizontalFlip(), 
                         transforms.RandomCrop(128, padding=4, padding_mode='reflect'), #transforms.RandomCrop(224, padding=4, padding_mode='reflect'), #RandomCrop(32, padding=4, padding_mode='reflect')
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
       
        transform_test = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor(), transforms.Normalize(*stats)]) #transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize(*stats)])
        """
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


def get_dataloader(dataset, train_batch_size, test_batch_size, root='./data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    # adapt root folder based on chosen dataset
    root = root + dataset + '_data/'
    # get dataset
    if dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'SVHN':
        trainset = torchvision.datasets.SVHN(root=root, split = 'train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=root, split = 'test', download=True, transform=transform_test)
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dataset == 'imagenet':
        trainset = torchvision.datasets.ImageNet(root=root, split = 'train', transform=transform_train)
        testset = torchvision.datasets.ImageNet(root=root, split = 'val', transform=transform_test)
    elif dataset == 'imagenette_fs_v2':
        trainset = torchvision.datasets.ImageFolder(root + 'train/', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + 'val/', transform=transform_test)
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
    
    #### get number of classes
    if dataset == 'SVHN': #SVHN object does not have a ".classes" attribute
        num_classes = 10
    else:
        num_classes = len(trainset.classes)
    return trainset, testset, num_classes
