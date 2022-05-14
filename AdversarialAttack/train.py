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
from PGD.pgd import PGD_TRAIN
from torch.utils.tensorboard  import SummaryWriter
from dataset import CIFAR10
from network import ConvNet, ConvNet2, ConvNet_quant
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
        # print(imgs.shape)
        # exit(123)
        output = model(imgs,args.quantization)
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

def pgd_train(epoch, model, optimizer, criterion, train_loader, writer, eps=8/255,
                 alpha=2/255, steps=4):
    model.train()
    

    # exit(123)
    pgd = PGD_TRAIN(model,eps,alpha)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        
        # print(imgs.shape)
        # exit(123)
        # print(model.conv2.weight)
        # print("???")
        # print(imgs.shape)
        # print(torch.abs(model.conv1.weight).shape)
        # print(                            torch.max(torch.abs(model.conv1.weight),dim = 1)[0].shape)
        # print(                  torch.max(torch.max(torch.abs(model.conv1.weight),dim = 1)[0],dim = 1)[0].shape)
        
            # exit(123)

        optimizer.zero_grad()
        output = model(imgs,args.quantization)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

        loss.backward()
        if args.args.quantization:
            for p,q in zip(model.conv1.parameters(),model.conv1_int.parameters()):
                p.grad = q.grad
            for p,q in zip(model.conv2.parameters(),model.conv2_int.parameters()):
                p.grad = q.grad
            for p,q in zip(model.Linear.parameters(),model.Linear_int.parameters()):
                p.grad = q.grad
        # model.conv1.grad = model.conv1_int.grad
        # model.conv2.grad = model.conv2_int.grad
        # model.Linear.grad = model.Linear_int.grad
        pgd.backup_grad()
        for t in range(steps):
            pgd.attack(is_first_attack=(t==0)) 
            if t != steps-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            output_adv = model(imgs,args.quantization)
            loss_adv = criterion(output_adv, labels)
            loss_adv.backward()
        pgd.restore()

        optimizer.step()
        model.zero_grad()

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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    print("Finish Loadding All Data ~")
    # define network args.
    if args.quant_model:
        model = ConvNet_quant()
    else:
        model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()    

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        if args.attack :
            pgd_train(epoch, model, optimizer, criterion, train_loader, writer) 
        else:      
            train(epoch, model, optimizer, criterion, train_loader, writer)
        
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() ,
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch, model, val_loader, writer)
    return 

# TODO: 训练与测试代码
if __name__ == '__main__':
    setup_seed(1)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-4, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=300, help="total epoch number for training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=200, help="batch size")

    arg_parser.add_argument('--cont', '-continue', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--attack', '-attack', action='store_true', help="whether to train with PGD")
    arg_parser.add_argument('--quant_model', '-quant_model', action='store_true', help="whether to use small model")
    arg_parser.add_argument('--quantization', '-quantization', action='store_true', help="whether to use small model")
    args = arg_parser.parse_args()

    run(args)