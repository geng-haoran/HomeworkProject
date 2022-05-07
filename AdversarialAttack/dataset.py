import os
import math
import numpy as np
import random

from attrdict import AttrDict
import pickle
from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
# 数据集
class CIFAR10():
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 10
    IMAGE_SIZE = [32, 32]
    IMAGE_CHANNELS = 3
    
    def __init__(self,train=True):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])
        self.train = train
        self.load_dataset(transform)
        self.train_imgs = []
        self.train_gts = []
        self.test_imgs = []
        self.test_gts = []

        for i in self.trainset:
            self.train_imgs.append(i[0])
            self.train_gts.append(i[1])
        for i in self.testset:
            self.test_imgs.append(i[0])
            self.test_gts.append(i[1])
        self.train_imgs = np.vstack(self.train_imgs).reshape(-1, 3, 32, 32)
        self.train_imgs = self.train_imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.test_imgs = np.vstack(self.test_imgs).reshape(-1, 3, 32, 32)
        self.test_imgs = self.test_imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.train_gts = np.array(self.train_gts)
        self.test_gts = np.array(self.test_gts)
        if self.train:
            self.data = self.train_imgs
            # print(self.data.shape)
            self.targets = self.train_gts
        else:
            self.data = self.test_imgs
            self.targets = self.test_gts

    
    def load_dataset(self, transform):
        self.trainset = torchvision.datasets.CIFAR10(root="./data", transform=transform, download=False)
        self.testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=False)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        # print(img.shape,target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = CIFAR10()
    print(len(dataset.trainset))
    print(len(dataset.testset))
    print(dataset.train_imgs.shape)
    print(dataset.test_gts.shape)