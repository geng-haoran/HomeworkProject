from dataclasses import dataclass
import torch
from utils import * 
import os,sys
import glob
from os.path import join as pjoin
import cv2
import numpy as np
import pickle
import random
import torchvision.transforms as tfs
from PIL import Image
# import torchvision.transforms as transforms
HEIGHT = 512
WIDTH = 512
def random_crop(image, crop_shape, padding=None):
    oshape = image.size

    if padding:
        oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
        img_pad.paste(image, (padding, padding))
        
        nh = random.randint(0, oshape_pad[0] - crop_shape[0])
        nw = random.randint(0, oshape_pad[1] - crop_shape[1])
        image_crop = img_pad.crop((nh, nw, nh+crop_shape[0], nw+crop_shape[1]))

        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image

# save_root = "/data2/haoran/HW/HomeworkProject/PlantSeedClassification/datav3"
class PlantSeed(torch.utils.data.Dataset):
    def __init__(self,train = True,few_shot = False,few_num = 256,final_test = False):
        super(PlantSeed,self).__init__()
        self.train = train
        self.data_root = "/data2/haoran/HW/HomeworkProject/PlantSeedClassification/datav3"
        self.train_dataname = np.array([k for j in (glob.glob(pjoin(self.data_root,"train",i,"*")) for i in LABELS) for k in j])
        self.train_label = np.array([NAME2LABEL[k.split("/")[-2]] for k in self.train_dataname])
        self.test_dataname = np.array([k for k in glob.glob(pjoin(self.data_root,"test","*"))])
        self.eval_dataname = np.array([k for j in (glob.glob(pjoin(self.data_root,"eval",i,"*")) for i in LABELS) for k in j])
        self.eval_label = np.array([NAME2LABEL[k.split("/")[-2]] for k in self.eval_dataname])
        self.final_test = final_test
        if final_test:
            print("final_test!",self.final_test)
        if few_shot:
            index = index = np.random.randint(0,len(self.train_dataname),few_num)
            self.train_dataname = self.train_dataname[index]
            self.train_label = self.train_label[index]
        if train:
            self.data = np.array([cv2.imread(k) for k in self.train_dataname])
            self.label = self.train_label
        if not train:
            self.data = np.array([cv2.imread(k) for k in self.eval_dataname])
            self.label = self.eval_label
        if final_test:
            self.data = np.array([cv2.imread(k) for k in self.test_dataname])
            self.name_list = np.array([i.split("/")[-1] for i in self.test_dataname])

    def __getitem__(self,index):
        if self.final_test:

            img,name = self.data[index],self.name_list[index]
            img = Image.fromarray(img)
            transform_GY = tfs.ToTensor()#将PIL.Image转化为tensor，即归一化。
            transform_BZ= tfs.Normalize(
                mean= [0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # transform_compose
            transform_compose= tfs.Compose([
                transform_GY ,
                transform_BZ])
            transform_compose(img)
            img = tfs.ToTensor()(img).float()
            return img,name
        img, label = self.data[index], self.label[index]
        #### Augmentation
        img = Image.fromarray(img)
        p = random.random() >= 0.5
        if self.train and p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train:
            img = tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1)(img)
            crop_width = img.size[0] - 24
            crop_height = img.size[1] - 24
            img = random_crop(img, [crop_width, crop_height], padding=10)

        transform_GY = tfs.ToTensor()#将PIL.Image转化为tensor，即归一化。

        transform_BZ= tfs.Normalize(
            mean= [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # transform_compose
        transform_compose= tfs.Compose([
            transform_GY ,
            transform_BZ])
        transform_compose(img)
        img = tfs.ToTensor()(img).float()

            
        ####
        # else:
        #     img = tfs.ToTensor()(img).float() * 255
        
        return img, label
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = PlantSeed(train = False,few_shot = True)

