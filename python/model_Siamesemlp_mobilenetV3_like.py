import os
import random
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, accuracy_score
import winsound
from torchvision import transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")
# transform = transforms.ToTensor()
def letterbox_gray_cv2(image, target_size=256):
    # 原始图像尺寸 (OpenCV是height, width顺序)
    h, w = image.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    
    # 等比例缩放后的新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像 (使用cv2.resize进行高质量重采样)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 创建一个目标尺寸的空白灰度图像
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 将缩放后的图像居中放置
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    
    return canvas

class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1,ReLU:bool=False,activate:bool=True):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if ReLU else nn.ReLU6(inplace=True)
        self.act = self.act if activate else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels,1),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
            scale = self.avg_pool(x)       # [B, C, 1, 1]
            scale = self.fc(scale)     # [B, C, 1, 1]
            return x * scale           # 逐通道缩放

class block(nn.Module):
    def __init__(self, in_channels: int,mid_channels: int, out_channels: int, stride: int = 1,kernel_size: int = 3,SE_flag: bool = False):
        super(block, self).__init__()
        self.shortcut = (in_channels == out_channels and stride == 1)
        self.depthwise_conv = nn.Sequential(
            ConvBNAct(in_channels,mid_channels,kernel_size=1),#点卷积，提升通道数
            ConvBNAct(mid_channels,mid_channels,kernel_size=kernel_size,groups=mid_channels,stride=stride),#分组卷积，提升感受野Relu6
            SE(mid_channels) if SE_flag else nn.Identity(),
            ConvBNAct(mid_channels,out_channels,kernel_size=1,activate=False)#卷积，降低通道数不激活
        )
    def forward(self, x):
        out = self.depthwise_conv(x)
        if self.shortcut:
            out += x
        return out
    

class MobilenetV3_like(nn.Module):
    def __init__(self,in_channels: int = 3, num_embeddings: int = 104):
        super(MobilenetV3_like, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.first_conv = ConvBNAct(in_channels, 16, 3, 1, 1)
        self.block1 = block(16,16,16,2,3,True)
        self.block2 = block(16,72,24,2,3,False)
        self.block3 = block(24,88,24,1,3,False)
        self.block4 = block(24,96,40,2,5,True)
        self.block5 = block(40,240,40,1,5,True)
        self.block6 = block(40,240,40,1,5,True)
        self.block7 = block(40,120,48,1,5,True)
        self.embedding = nn.Sequential(
            nn.Linear(48,480),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(480,num_embeddings)
        )
    def forward(self, x):
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return x
class SiameseNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_embeddings: int = 104):
        super(SiameseNet, self).__init__()
        self.mobilenetv3_like = MobilenetV3_like(in_channels, num_embeddings)

    def forward(self, x1, x2):
        x1 = self.mobilenetv3_like(x1)
        x2 = self.mobilenetv3_like(x2)
        return x1, x2