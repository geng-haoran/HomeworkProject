import os
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

from torch.utils.tensorboard  import SummaryWriter
from dataset import CIFAR10
from network import ConvNet
###################
    # 设置随机种子，保证实验可复现性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, model, val_loader, writer):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        bsz = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

    writer.add_scalar('val/top@1', top1.avg, epoch)
    writer.add_scalar('val/top@5', top5.avg, epoch)

    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def validate_attack(epoch, model, val_loader, writer):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print(val_loader)
    exit(123)
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        bsz = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

    writer.add_scalar('val/top@1', top1.avg, epoch)
    writer.add_scalar('val/top@5', top5.avg, epoch)

    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 


def train(epoch, model, optimizer, criterion, train_loader, writer):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output = model(imgs)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0: 
            writer.add_scalar('train/loss', loss, iteration)
            writer.add_scalar('train/top@1', top1.avg, iteration)
            writer.add_scalar('train/top@5', top5.avg, iteration)

    print(' Epoch: %d'%(epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def run(args):
    save_folder = os.path.join('../experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    train_dataset = CIFAR10()
    val_dataset = CIFAR10(train=False)
    attack_dataset = CIFAR10(train = False, attack = True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    print("Finish Loadding All Data ~")
    # define network 
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()    

    # load latest checkpoint
    read_path = ckpt_path
    print('load checkpoint from %s'%(read_path))
    checkpoint = torch.load(read_path)
    model.load_state_dict(checkpoint['model'])


    validate_attack(epoch, model, val_attack_loader, writer)
            # validate_attack(epoch, model, val_loader, writer)
    return 

# TODO: 训练与测试代码
if __name__ == '__main__':
    setup_seed(1)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--ckpt_path', '-p', type=str, required=True, help="The checkpoints to be attacked")
    args = arg_parser.parse_args()

    run(args)