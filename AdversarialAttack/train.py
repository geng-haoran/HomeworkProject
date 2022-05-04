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

###################3

# 数据集
class CIFAR10():
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 10
    IMAGE_SIZE = [32, 32]
    IMAGE_CHANNELS = 3
    
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])
        self.load_dataset(transform)
    
    def load_dataset(self, transform):
        self.trainset = torchvision.datasets.CIFAR10(root="./data", transform=transform, download=True)
        self.testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

# 设置随机种子，保证实验可复现性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DynamicAct(nn.Module):
    # TODO: 实现动态阈值的激活函数
    pass

# TODO: 搭建卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__()
        
    def forward(self, x, quant=False):
        """
            x: 输入图片
            quant: 是否使用模型量化
        """
        pass
    
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

# TODO: 训练与测试代码
setup_seed(1)