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

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p=0.2)

    def forward_once(self, x):
        x_hu = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = torch.cat((x_hu, x), dim=1)
        return x

    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)

class TripletNet_bn2(nn.Module):
    def __init__(self):
        super(TripletNet_bn2, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(7)
        self.bn2 = nn.BatchNorm1d(32)
        # self.bn3 = nn.BatchNorm1d(23)

    def forward_once(self, x):
        x_hu = x
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.cat((x_hu, x), dim=1)
        # x = self.bn3(x)
        return x

    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)
# ======= MLP 网络结构 =======
class TripletNet_attention(nn.Module):
    def __init__(self):
        super(TripletNet_attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(7,7),
            nn.Softmax(dim=-1)
        )
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(7)
        self.bn2 = nn.BatchNorm1d(32)
        # self.bn3 = nn.BatchNorm1d(23)

    def forward_once(self, x):
        x_hu=x
        x = self.bn1(x)
        weights = self.attention(x_hu)
        x_hu= torch.mul(x_hu, weights)+x_hu
       
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.cat((x_hu, x), dim=1)
        return x

    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)

class HuNet_1D(nn.Module):
    """Hu‑矩差(1) + 16D 差 → 17D → 1D 预测"""
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.head = nn.Linear(17, 1)          # 最终 1 个神经元

    def _embed16(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)                    # (B,16)

    def forward(self, hu_a, hu_b):
        d_hu = torch.sum(torch.abs(hu_b - hu_a), 1, keepdim=True)  # (B,1)
        d16  = self._embed16(hu_b) - self._embed16(hu_a)           # (B,16)
        feats = torch.cat([d_hu, d16], 1)                          # (B,17)
        return nn.Sigmoid(self.head(feats))                                   # (B,1)

class HuNetAttn_1D(nn.Module):
    def __init__(self, channel=7, reduction=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.BatchNorm1d(channel),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channel // reduction),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.bn0 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 32, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.head = nn.Linear(17, 1)

    def _embed16(self, x_raw):
        x= x_raw
        x = self.bn0(x)
        x= self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        return self.fc2(x)

    def forward(self, hu_a, hu_b):
        d16 = self._embed16(hu_b) - self._embed16(hu_a)
        w_a = self.attn(hu_a)
        w_b = self.attn(hu_b)
        hu_a = hu_a + hu_a * w_a
        hu_b = hu_b + hu_b * w_b
        d_hu = torch.sum(torch.abs(hu_b - hu_a), 1, keepdim=True)
        feats = torch.cat([d_hu, d16], 1)
        return torch.sigmoid(self.head(feats))

class HuNetAttn_Triplet(nn.Module):
    """anchor / pos / neg → 三个 17D 向量"""
    def __init__(self, channel=7, reduction=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.bn0 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)

    def _embed16(self, x_raw):
        x = x_raw + x_raw * self.attn(x_raw)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

    def forward_once(self, hu):
        return torch.cat([
            torch.sum(torch.abs(hu - hu.mean(0, keepdim=True)), 1, keepdim=True),  # 1D: 与批均 Hu 差，可改为恒 0
            self._embed16(hu)
        ], 1)  # (B,17)

    def forward(self, anchor, positive, negative):
        f_anchor   = self.forward_once(anchor)
        f_positive = self.forward_once(positive)
        f_negative = self.forward_once(negative)
        return f_anchor, f_positive, f_negative
