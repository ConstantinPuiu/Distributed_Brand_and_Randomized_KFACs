import torch.nn as nn
import torch.nn.functional as F

class FC_net_for_CIFAR10(nn.Module):
    def __init__(self, num_classes):
        super(FC_net_for_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(1024, 2048) # nn.Linear(7 * 16, 30)
        self.fc2 = nn.Linear(2048, 8192) # nn.Linear(30, 10)
        self.fc3 = nn.Linear(8192, 8192)
        self.fc4 = nn.Linear(16384, 16384)
        self.fc5 = nn.Linear(16384, 32768)
        self.fc6 = nn.Linear(32768, 65536)
        self.fc7 = nn.Linear(65536, 16384)
        self.classifier = nn.Sequential(
                                              nn.Linear(512 * 8 * 4, 512 * 4),
                                              nn.ReLU(True),
                                              nn.Linear(512 * 4, num_classes),
                                          )

    def features(self, x, nodes_dropout=False):
        x = x.view(-1, 1024) # 32*32 = 1024
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        return x

    def forward(self, x, nodes_dropout=False):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    