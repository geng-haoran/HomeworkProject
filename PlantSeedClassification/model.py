import torch.nn as nn
from utils import *
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
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, 3),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(15625*16, 128),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(1)
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