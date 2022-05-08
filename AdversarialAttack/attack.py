import os
import sys
from os.path import join as pjoin
import math
import numpy as np
import random
import tqdm 
from attrdict import AttrDict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from util import evaluate, AverageMeter
import torchvision
import torchvision.transforms as transforms
from PGD.pgd import PGD
from dataset import CIFAR10
from network import ConvNet
import cv2
###################
visu_root = "/data2/haoran/HW/HomeworkProject/AdversarialAttack/visu"

    # 设置随机种子，保证实验可复现性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate_attack(model, val_loader, attack = False):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # print(val_loader)
    # exit(123)
    preds = []
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        # print(imgs)
        # exit(123)
        bsz = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()

        if not attack:
            topk = (1,)
            maxk = max(topk)
            batch_size = labels.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            
            preds.append(list(pred.reshape(-1)))
            
            # cv2.imwrite(pjoin(visu_root,f"test{0}"+".png"),(imgs[0]*255).astype(np.uint8))

        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)
    # if not attack:
    #     print(preds)
    #     print(torch.tensor(preds))
    #     print(torch.tensor(preds).reshape(-1))
    #     print(torch.tensor(preds).reshape((-1)).shape)
    #     np.save("/data2/haoran/HW/HomeworkProject/AdversarialAttack/result/raw_pred.npy",torch.tensor(preds).reshape((-1)).numpy(), allow_pickle='TRUE')
    #     exit(123)
    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def run(args):
    save_folder = os.path.join('../experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    
    # define network 
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define loss
    criterion = torch.nn.CrossEntropyLoss()    

    # load latest checkpoint
    read_path = args.ckpt_path
    print('load checkpoint from %s'%(read_path))
    checkpoint = torch.load(read_path)
    model.load_state_dict(checkpoint['model'])

    # define dataset and dataloader
    val_dataset = CIFAR10(attack = True, model = model)
    print(val_dataset.raw_test_imgs.shape)
    print(val_dataset.raw_test_imgs.max(0).max(0).max(0))
    print(val_dataset.raw_test_imgs.min(0).min(0).min(0))
    print(val_dataset.raw_adv_images.shape)
    print(val_dataset.raw_adv_images.max(0).max(0).max(0))
    print(val_dataset.raw_adv_images.min(0).min(0).min(0))
    # visu_path = pjoin(visu_root,f"adv{1}"+".png")

    ######################  visualization  ######################
    # for i in range(10):
    #     cv2.imwrite(pjoin(visu_root,f"t_raw{i}"+".png"),(val_dataset.raw_test_imgs[i]).astype(np.uint8))
    #     cv2.imwrite(pjoin(visu_root,f"t_adv{i}"+".png"),(255*val_dataset.raw_adv_images[i]).astype(np.uint8))
    # exit(123)
    ######################  visualization  ######################

    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    # print("Finish Loadding All Data ~")
    print("Attacked !!!")
    validate_attack(model, val_loader, attack = True)

    val_dataset = CIFAR10(train = False)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    # print("Finish Loadding All Data ~")
    
    print("Not Attacked")

    validate_attack(model, val_loader)
    return 

# TODO: 训练与测试代码
if __name__ == '__main__':
    setup_seed(1)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--ckpt_path', '-p', type=str, required=True, help="The checkpoints to be attacked")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=20, help="batch size")   
    args = arg_parser.parse_args()

    run(args)