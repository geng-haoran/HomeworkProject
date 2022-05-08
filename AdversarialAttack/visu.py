import numpy as np
import torch
import cv2
from PGD.pgd import PGD
from dataset import CIFAR10
from network import ConvNet
from util import *
from os.path import join as pjoin
ckpt_path = "/data2/haoran/HW/HomeworkProject/experiments/just_new/ckpt/ckpt_epoch_61"
visu_root = "/data2/haoran/HW/HomeworkProject/AdversarialAttack/visu"
raw_pred = np.load("/data2/haoran/HW/HomeworkProject/AdversarialAttack/result/raw_pred.npy",allow_pickle='TRUE')
attack_pred = np.load("/data2/haoran/HW/HomeworkProject/AdversarialAttack/result/attack_pred.npy",allow_pickle='TRUE')

count = 0
# define network 
model = ConvNet()
if torch.cuda.is_available():
    model = model.cuda()

# load latest checkpoint
read_path = ckpt_path
print('load checkpoint from %s'%(read_path))
checkpoint = torch.load(read_path)
model.load_state_dict(checkpoint['model'])
val_dataset = CIFAR10(attack = True, model = model)
for i in range(100):
    if raw_pred[i] != attack_pred[i] and raw_pred[i] == val_dataset.raw_test_gts[i]:
        img = np.ones((440,1000, 3), dtype=np.uint8) * 255
        img[50:370,100:420] = cv2.resize((val_dataset.raw_test_imgs[i]).astype(np.uint8), (320,320), interpolation = cv2.INTER_AREA)
        img[50:370,580:900] = cv2.resize((255*val_dataset.raw_adv_images[i]).astype(np.uint8), (320,320), interpolation = cv2.INTER_AREA)
        cv2.putText(img,LABEL2NAME[raw_pred[i]],(200,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(img,LABEL2NAME[attack_pred[i]],(700,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
        cv2.imwrite(pjoin(visu_root,f"result{count}_{i}"+".png"),img[...,::-1])
        count += 1
