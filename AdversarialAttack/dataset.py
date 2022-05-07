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
from PGD.pgd import PGD
import torchvision
import torchvision.transforms as transforms
# 数据集
class CIFAR10():
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 10
    IMAGE_SIZE = [32, 32]
    IMAGE_CHANNELS = 3
    
    def __init__(self,train=True,attack = False,model=None, eps=8/255, alpha=2/255, steps=4):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])
        self.train = train
        self.load_dataset(transform)
        self.train_imgs = []
        self.train_gts = []
        self.test_imgs = []
        self.test_gts = []
        self.raw_test_imgs = []
        self.raw_test_gts = []

        for i in self.trainset:
            self.train_imgs.append(i[0])
            self.train_gts.append(i[1])
        for i in self.testset:
            self.test_imgs.append(i[0])
            self.test_gts.append(i[1])
        for i in self.raw_testset:
            self.raw_test_imgs.append(i[0])
            self.raw_test_gts.append(i[1])
        self.train_imgs = np.array([np.array(i) for i in self.train_imgs])
        self.test_imgs = np.array([np.array(i) for i in self.test_imgs])
        self.raw_test_imgs = np.array([np.array(i) for i in self.raw_test_imgs])
        self.train_gts = np.array(self.train_gts)
        self.test_gts = np.array(self.test_gts)
        self.raw_test_gts = np.array(self.raw_test_gts)
        if self.train:
            self.data = self.train_imgs.transpose((0,2,3,1))
            self.targets = self.train_gts
        else:
            
            self.data = self.test_imgs.transpose((0,2,3,1))
            # print(self.data.shape)
            self.targets = self.test_gts
        if attack:
            atk = PGD(model=model, eps=eps, alpha=alpha, steps=steps)
            # for i in range(self.raw_test_imgs):
            # print(torch.from_numpy(self.raw_test_imgs).shape)
            # print(torch.from_numpy(self.raw_test_imgs.transpose((0,3,1,2))).shape)
            self.adv_images = atk(torch.from_numpy(self.raw_test_imgs.transpose((0,3,1,2))).float()/255, torch.from_numpy(self.raw_test_gts))
            self.raw_adv_images = self.adv_images.cpu().numpy().transpose((0,2,3,1))
            
            self.data = self.adv_images.cpu().numpy().transpose((0,2,3,1))
            self.targets = self.raw_test_gts
            for i in range(self.data.shape[0]):
                # print(transform(self.data[i].copy()).shape)
                self.data[i] = transform(self.data[i].copy()).numpy().transpose((1,2,0))
            # print(self.adv_images)
            # print(self.data.shape)
            # exit(123)
            # print(self.targets.shape)
            # print(self.data.max(0).max(0).max(0))
            # print(self.data.min(0).min(0).min(0))
        
        # print(self.test_imgs.max(0).max(0).max(0))
        # print(self.test_imgs.min(0).min(0).min(0))
        # print((self.raw_test_imgs).max(0).max(0).max(0))
        # print((self.raw_test_imgs).min(0).min(0).min(0))
        # print(self.raw_test_imgs.shape)
        # exit(123)

    
    def load_dataset(self, transform):
        self.trainset = torchvision.datasets.CIFAR10(root="./data", transform=transform, download=False)
        self.testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=False)
        self.raw_testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=None, download=False)
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