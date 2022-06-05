import os,sys,torch,numpy as np
from re import I
from os.path import join as pjoin
import glob

from tqdm import tqdm
from utils import *
from dataset import PlantSeed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import AverageMeter, evaluate
import tqdm
import argparse
import torch.optim as optim
from model import *

def train(epoch, model, optimizer, criterion, train_loader, writer):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        batch_size = labels.shape[0]
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        output = model(imgs)
        loss = criterion(output, labels)

        losses.update(loss.item(), batch_size)
        acc1, acc3, acc5 = evaluate(output, labels,(1,3,5))
        top1.update(acc1.item(), batch_size)
        top3.update(acc3.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('train/loss', loss, iteration)
            writer.add_scalar('train/top@1', top1.avg, iteration)
            writer.add_scalar('train/top@3', top3.avg, iteration)
            writer.add_scalar('train/top@5', top5.avg, iteration)
    print(' Epoch: %d'%(epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@3 {top3.avg:.3f}'.format(top3=top3))
    print(' Train Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def validate(epoch, model, val_loader, writer):
    model.eval()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        batch_size = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
        acc1, acc3, acc5 = evaluate(output, labels, topk=(1, 3, 5))
        top1.update(acc1.item(), batch_size)
        top3.update(acc3.item(), batch_size)
        top5.update(acc5.item(), batch_size)

    writer.add_scalar('val/top@1', top1.avg, epoch)
    writer.add_scalar('val/top@3', top3.avg, epoch)
    writer.add_scalar('val/top@5', top5.avg, epoch)


    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@1 {top3.avg:.3f}'.format(top3=top3))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_root', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-3, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=100, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=50, help="batch size")
    arg_parser.add_argument('--optimizer', '-o', type=str, default="Adam", help="optimizer type: Adam SGD Adagrad")
    arg_parser.add_argument('--model', '-m', type=str, default="resnet18", help="model: resnet18 conv vgg16 MLP")
    arg_parser.add_argument('--few_shot', '-f', action='store_true', help="only use few data")

    args = arg_parser.parse_args()

    save_folder = pjoin("/data2/haoran/HW/HomeworkProject/PlantSeedClassification/exp",args.exp_root)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)
    if args.few_shot:
        train_dataset = PlantSeed(few_shot=True)
    else:
        train_dataset = PlantSeed()
    print("Finish Loading All Data~")
    val_dataset = PlantSeed(train = False)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle = True, num_workers = 10, drop_last=True,)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle = False, num_workers = 10)
    
    if args.model == "resnet18":
        model = ResNet18()
    elif args.model == "vgg16":
        model = VGG16()
    elif args.model == "conv":
        model = ConvNet()
    elif args.model == "MLP":
        model = MLPNet()
    elif args.model == "conv_nonorm":
        model = ConvNet_noNorm
    elif args.model == "MLP_norm":
        model = MLPNet_Norm()
    elif args.model == "MLP_NormDropout":
        model = MLPNet_NormDropout()
    else:
        print("No such model")
        exit(0)
    model = ResNet18()
    if torch.cuda.is_available():
        model = model.cuda()
    print("Finish Loading Model~")
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=0, weight_decay=1e-4, initial_accumulator_value=0)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, dampening=0, weight_decay=1e-4, nesterov=False)
    
    else:
        print('No such optimizer !')
        exit(123)

    criterion = torch.nn.CrossEntropyLoss()

    if args.cont:
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
        train(epoch, model, optimizer, criterion, train_loader, writer)
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch, model, val_loader, writer)
