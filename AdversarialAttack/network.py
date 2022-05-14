import os
import math
import numpy as np
import random

from attrdict import AttrDict
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard  import SummaryWriter
from dataset import CIFAR10
# from network import ConvNet

class DynamicAct(nn.Module):
    # TODO: 实现动态阈值的激活函数
    pass



# TODO: 搭建卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, num_class=10, **kwargs):
        super(ConvNet, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*25, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_class)
        )
        
    def forward(self, x, quant=False):
        """
            x: 输入图片
            quant: 是否使用模型量化
        """
        # print(x.shape)
        # exit(123)
        x = self.convlayer(x)
        return x
    
    # TODO: 计算正则项
    def regularizationTerm(self, reg_type):
        """
            reg_type: orthogonal 正交正则项; spectral 谱范数正则项
        """
        term = 0.0
        if reg_type == "orthogonal":
            pass
        elif reg_type == "spectral":
            pass
        else:
            raise NotImplementedError
        
        return term

class ConvNet2(nn.Module):
    def __init__(self, num_class=10, **kwargs):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.ReLU = nn.ReLU(inplace=True)
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(28*28*32, num_class)
        
    def forward(self, x, quant=False):
        """
            x: 输入图片
            quant: 是否使用模型量化
        """
        x1 = self.conv1(x)
        x2 = self.ReLU(x1)
        x3 = self.conv2(x2)
        x4 = self.ReLU(x3)
        x5 = self.Flatten(x4)
        x6 = self.Linear(x5)
        return x6      

    
    # TODO: 计算正则项
    def regularizationTerm(self, reg_type):
        """
            reg_type: orthogonal 正交正则项; spectral 谱范数正则项
        """
        term = 0.0
        if reg_type == "orthogonal":
            pass
        elif reg_type == "spectral":
            pass
        else:
            raise NotImplementedError
        
        return term

class ConvNet_quant(nn.Module):
    def __init__(self, num_class=10, **kwargs):
        super(ConvNet_quant, self).__init__()
        # self.convlayer = nn.Sequential(
        #     nn.Conv2d(3, 32, 3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2),
        #     # nn.Conv2d(32, 64, 3),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(64, 64, 3),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     # nn.Linear(64*25, 512),
        #     # nn.Dropout(p=0.5),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(28*28*32, num_class)
        # )
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv1_int = nn.Conv2d(3, 32, 3)
        # print(self.conv1.weight)
        # self.conv1.weight = self.conv1.weight * 2
        self.torch.ones((3),requires_grad = True)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv2_int = nn.Conv2d(32, 32, 3)
        self.ReLU = nn.ReLU(inplace=True)
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(28*28*32, num_class)
        self.Linear_int = nn.Linear(28*28*32, num_class)
    def myReLU(self, t, x):
        mask1 = x<t[0]
        mask2 = torch.logical_and(x>t[0],x<t[1])
        mask3 = torch.logical_and(x>t[1],x<t[2])
        mask4 = x>t[2]
        x[mask1] = 0
        x[mask2] = t[0]
        x[mask3] = t[1]
        x[mask4] = t[2]
        return x,mask1,mask2,mask3,mask4
    def forward(self, x,quant=True):
        """
            x: 输入图片
            quant: 是否使用模型量化
        """
        # print(quant)
        # if quant:
        #     with torch.no_grad():
        
        if quant:
            conv1_scale = (torch.max(torch.max(torch.max(torch.abs(self.conv1.weight),dim = 1)[0],dim = 1)[0],dim = 1)[0]/(2**(8-1)-1)).reshape((-1,1,1,1))
            self.conv1_int.weight = torch.nn.Parameter(((self.conv1.weight/conv1_scale).int()*conv1_scale).float())
            conv2_scale = (torch.max(torch.max(torch.max(torch.abs(self.conv2.weight),dim = 1)[0],dim = 1)[0],dim = 1)[0]/(2**(8-1)-1)).reshape((-1,1,1,1))
            self.conv2_int.weight = torch.nn.Parameter(((self.conv2.weight/conv2_scale).int()*conv2_scale).float())
            Linear_scale = torch.max(torch.abs(self.Linear.weight))/(2**(8-1)-1)
            self.Linear_int.weight = torch.nn.Parameter(((self.Linear.weight/Linear_scale).int()*Linear_scale).float())
                

            x1 = self.conv1(x)
            x2,mask11,mask12,mask13,mask14 = self.myReLU(self.t,x1)
            # x2 = self.ReLU(x1)
            x3 = self.conv2(x2)
            x4,mask21,mask22,mask23,mask24 = self.myReLU(self.t,x3)
            # x4 = self.ReLU(x3)
            x5 = self.Flatten(x4)
            x6 = self.Linear(x5)
            x1_int = self.conv1_int(x)
            x2_int = self.ReLU(x1_int)
            x3_int = self.conv2_int(x2_int)
            x4_int = self.ReLU(x3_int)
            x5_int = self.Flatten(x4_int)
            x6_int = self.Linear_int(x5_int)
            # x6.grad = x6_int.grad
            return x6_int,mask11,mask12,mask13,mask14,mask21,mask22,mask23,mask24
        else:
            print("no quantization")
            x1 = self.conv1(x)
            x2 = self.ReLU(x1)
            x3 = self.conv2(x2)
            x4 = self.ReLU(x3)
            x5 = self.Flatten(x4)
            x6 = self.Linear(x5)
            return x6

        # exit(123)
        
    
    # TODO: 计算正则项
    def regularizationTerm(self, reg_type):
        """
            reg_type: orthogonal 正交正则项; spectral 谱范数正则项
        """
        term = 0.0
        if reg_type == "orthogonal":
            pass
        elif reg_type == "spectral":
            pass
        else:
            raise NotImplementedError
        
        return term

# TODO: PGD 对抗攻击
class PGD():
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=4):
        pass
    
    def forward(self, images, labels):
        pass


if __name__ == "__main__":
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10() #.train_imgs
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        print(imgs[0].shape)
        writer.add_graph(net, imgs)
        writer.close()
        break 