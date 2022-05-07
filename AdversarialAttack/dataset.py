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
            print(self.data.shape)
            self.targets = self.train_gts
        else:
            self.data = self.test_imgs
            self.targets = self.test_gts

    
    def load_dataset(self, transform):
        self.trainset = torchvision.datasets.CIFAR10(root="./data", transform=transform, download=True)
        self.testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
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
        print(img.shape,target)

        return img, target

    def __len__(self):
        return len(self.data)

# class CIFAR10(torch.utils.data.Dataset):
#     """
#         modified from `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#     """
#     def __init__(self, train=True):
#         super(CIFAR10, self).__init__()

#         self.base_folder = '../datasets/cifar-10-batches-py'
#         self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4','data_batch_5']
#         self.test_list = ['test_batch']

#         self.meta = {
#             'filename': 'batches.meta',
#             'key': 'label_names'
#         }

#         self.train = train  # training set or test set
#         if self.train:
#             file_list = self.train_list
#         else:
#             file_list = self.test_list

#         self.data = []
#         self.targets = []

#         # now load the picked numpy arrays
#         for file_name in file_list:
#             file_path = os.path.join(self.base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 entry = pickle.load(f, encoding='latin1')
#                 self.data.append(entry['data'])
#                 if 'labels' in entry:
#                     self.targets.extend(entry['labels'])
#                 else:
#                     self.targets.extend(entry['fine_labels'])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

#         self._load_meta()

#     def _load_meta(self):
#         path = os.path.join(self.base_folder, self.meta['filename'])
#         with open(path, 'rb') as infile:
#             data = pickle.load(infile, encoding='latin1')
#             self.classes = data[self.meta['key']]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]
#         img = img.astype(np.float32)
#         img = img.transpose(2, 0, 1)
        
#         # ------------TODO--------------
#         # data augmentation
#         # img = Image.fromarray(img)
#         # import random
#         # p = random.random() >= 0.5
#         # if self.train and p:
#         #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         # if self.train:
#         #     img = tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1)(img)
#         # img = tfs.ToTensor()(img).float() * 255
#         # ------------TODO--------------

#         return img, target

#     def __len__(self):
#         return len(self.data)

# if __name__ == '__main__':
#     from PIL import Image
#     # --------------------------------
#     # The resolution of CIFAR-10 is tooooo low
#     # You can use Lenna.png as an example to visualize and check your code.
#     # Submit the origin image "Lenna.png" as well as at least two augmented images of Lenna named "Lenna_aug1.png", "Lenna_aug2.png" ...
#     # --------------------------------

#     # # Visualize CIFAR-10. For someone who are intersted.
#     # train_dataset = CIFAR10()
#     # i = 0
#     # for imgs, labels in train_dataset:
#     #     imgs = imgs.transpose(1,2,0)
#     #     cv2.imwrite(f'aug1_{i}.png', imgs)
#     #     i += 1
#     #     if i == 10:
#     #         break 

#     # Visualize and save for submission
#     img = Image.open('Lenna.png')
#     img.save('../results/Lenna.png')

#     # --------------TODO------------------
#     # Copy the first kind of your augmentation code here
#     # --------------TODO------------------
#     aug1 = img.transpose(Image.FLIP_LEFT_RIGHT)
#     aug1.save(f'../results/Lenna_aug1.png')

#     # --------------TODO------------------
#     # Copy the second kind of your augmentation code here
#     # --------------TODO------------------
#     aug2 = tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1)(img)
#     aug2.save(f'../results/Lenna_aug2.png')

if __name__ == "__main__":
    dataset = CIFAR10()
    print(len(dataset.trainset))
    print(len(dataset.testset))
    print(dataset.train_imgs.shape)
    print(dataset.test_gts.shape)