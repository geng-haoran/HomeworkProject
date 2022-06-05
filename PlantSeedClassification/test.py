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

def validate(model, test_loader):
    model.eval()
    out = {}
    for imgs,name in tqdm.tqdm(test_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
            out[name[0]] = np.argmax(output.numpy())
            # out.append(output.numpy())
    return out

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--pretrain', '-p', type=str, required=True, help="Pretrain Model")
    arg_parser.add_argument('--model', '-m', type=str, default="resnet18", help="model: resnet18 conv vgg16 MLP")
    args = arg_parser.parse_args()

        # train_dataset = PlantSeed()
    
    # test_dataset = PlantSeed(train = False)
    test_dataset = PlantSeed(final_test=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle = False, num_workers = 10)
    print("Finish Loading All Data~")
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
    
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint['model'])
    print("Finish Loading Model~")
    with torch.no_grad():
        out = validate(model, test_loader)
    np.save("/data2/haoran/HW/HomeworkProject/PlantSeedClassification/test_result.npy",np.array(out),allow_pickle=True)
