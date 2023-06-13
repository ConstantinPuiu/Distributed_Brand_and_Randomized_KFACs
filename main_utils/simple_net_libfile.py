import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, nodes_dropout=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 7, kernel_size=5)
        if nodes_dropout == True:
            self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7 * 16, 30) # nn.Linear(7 * 16, 30)
        self.fc2 = nn.Linear(30, 10) # nn.Linear(30, 10)
        #initialize parameters structure to map gradients from parameter format to math format and viceversa
        self.parameter_structure = [] #[i.shape for i in self.parameters()]
        self.coarser_param_structure = []
        for item in self.parameters():
            #print('Parameter on device {}'.format(item.device))
            item = item.shape
            number_of_elements = 1
            current_list = []
            for j in item:
                number_of_elements = number_of_elements * j
                current_list.append(j)
            if len(current_list) == 1:
                current_list.append(1)
            self.parameter_structure.append(current_list)
            self.coarser_param_structure.append(number_of_elements)
        self.even_coarser_param_structure = list(np.array([self.coarser_param_structure[i] for i in range(len(self.coarser_param_structure)) if i % 2 == 1]) + np.array([self.coarser_param_structure[i] for i in range(len(self.coarser_param_structure)) if i % 2 == 0]))
        self.number_of_parameters = np.sum(self.coarser_param_structure)

    def forward(self, x, nodes_dropout=False):
        try:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
        except:
            print('\ntype(x)={}, x:{}\n'.format(type(x),x))
            print('\nself.conv1(x): {}'.format(self.conv1(x)))
        if nodes_dropout == True:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 112)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

