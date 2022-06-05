import torch.nn as nn
from utils import *
import functools
import torchvision.models as models
class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout = False, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if dropout:
                modules.append(nn.Dropout(p=0.5))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class ConvNet_noNorm(nn.Module):
    def __init__(self, num_class = LABEL_NUM):
        super(ConvNet_noNorm,self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(250000, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class)
        )
    def forward(self,x):
        return self.convlayer(x)

class ConvNet(nn.Module):
    def __init__(self, num_class = LABEL_NUM):
        super(ConvNet,self).__init__()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(15625*16, 128),
            nn.Dropout(p=0.5),
            norm_fn(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class)
        )
        
    def forward(self,x):
        return self.convlayer(x)

class MLPNet(nn.Module):
    def __init__(self, num_class = LABEL_NUM):
        super(MLPNet,self).__init__()
        self.mlp = MLP(3, num_class)
class MLPNet_Norm(nn.Module):
    def __init__(self, num_class  = LABEL_NUM):
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        super(MLPNet_Norm,self).__init__()
        self.mlp = MLP(3, num_class, norm_fn)

class MLPNet_NormDropout(nn.Module):
    def __init__(self, num_class  = LABEL_NUM):
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        super(MLPNet_NormDropout,self).__init__()
        self.mlp = MLP(3, num_class,True, norm_fn)

class ResNet18(nn.Module):
    def __init__(self,num_class = LABEL_NUM):
        super(ResNet18,self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_class)

    def forward(self,x):
        return self.resnet18(x)


class VGG16(nn.Module):
    def __init__(self,num_class = LABEL_NUM):
        super(VGG16,self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs,num_class)


    def forward(self,x):
        return self.vgg16(x)
