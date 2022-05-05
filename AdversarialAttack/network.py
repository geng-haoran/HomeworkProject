import os
import math
import numpy as np
import random

from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard  import SummaryWriter

class DynamicAct(nn.Module):
    # TODO: 实现动态阈值的激活函数
    pass

# TODO: 搭建卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, num_class=10, **kwargs):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
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
        x = self.layer1(x)
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



# TODO: PGD 对抗攻击
class PGD():
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=4):
        pass
    
    def forward(self, images, labels):
        pass
