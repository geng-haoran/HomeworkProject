import torch.nn as nn
from utils import *
import functools
class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class ConvNet_noNorm(nn.Module):
    def __init__(self, num_class = LABEL_NUM):
        super(ConvNet_noNorm,self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(250000, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class)
        )
    def forward(self,x):
        return self.convlayer(x)

class ConvNet(nn.Module):
    def __init__(self, num_class = LABEL_NUM):
        super(ConvNet,self).__init__()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(15625*16, 128),
            nn.Dropout(p=0.5),
            norm_fn(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class)
        )
        
        # self.max_pool = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten(2)
        # self.linear = nn.Linear(15625, 128)
        # self.dpout = nn.Dropout(p=0.5)
        # # nn.BatchNorm1d(16)
        # self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(128, num_class)
        # nn.Sigmoid()
        
    def forward(self,x):
        # x = self.convlayer(x)
        # print(x.shape)
        # x = self.max_pool(x)
        # print(x.shape)
        # x = self.flatten(x)
        # print(x.shape)
        # exit(123)

        return self.convlayer(x)