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
from data import CIFAR10
###################


# 设置随机种子，保证实验可复现性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# TODO: 训练与测试代码
setup_seed(1)


if __name__ == "__main__":
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        print(imgs[0].shape)
        writer.add_graph(net, imgs)
        writer.close()
        break 