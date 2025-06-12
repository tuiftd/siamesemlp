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
import matplotlib.pyplot as plt
import gc
from collections import deque
import statistics
import torch

gc.collect()
torch.cuda.empty_cache() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
transform = transforms.ToTensor()


def plot_pos_neg_dist(
    pos_dists,
    neg_dists,
    epoch,
    margin=None,
    best_threshold=None,
    save_dir="dist_plots",
    bins=50,
):
    """
    绘制当前 epoch 的正负样本距离分布图
    """
    # Ensure list or numpy for compatibility
    if isinstance(pos_dists, torch.Tensor):
        pos_dists = pos_dists.detach().cpu().numpy()
    if isinstance(neg_dists, torch.Tensor):
        neg_dists = neg_dists.detach().cpu().numpy()
    pos_dists = np.concatenate(pos_dists)
    neg_dists = np.concatenate(neg_dists)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(pos_dists, bins=bins, alpha=0.6, label="Positive Distance", color="green", density=True)
    plt.hist(neg_dists, bins=bins, alpha=0.6, label="Negative Distance", color="red", density=True)

    # Draw margin line
    if margin is not None:
        plt.axvline(x=margin, color="blue", linestyle="--", linewidth=2, label=f"Margin = {margin:.2f}")

    # Draw best F1 threshold
    if best_threshold is not None:
        plt.axvline(x=best_threshold, color="purple", linestyle=":", linewidth=2, label=f"Best F1 Threshold = {best_threshold:.2f}")

    # Labels
    plt.title(f"Distance Distributions at Epoch {epoch}")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    # Save
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()

def remove_outliers(data, method='iqr', threshold=1.5):
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        mask = (data >= q1 - threshold * iqr) & (data <= q3 + threshold * iqr)
        return data[mask]
    else:
        raise NotImplementedError(f"Outlier removal method '{method}' not implemented.")

def blockwise_morph_noise(image, block_size=16, p_dilate=0.3, p_erode=0.3):
    h, w = image.shape
    noisy = image.copy()
    shape_choice = random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE])
    kernel = cv2.getStructuringElement(shape_choice, (3, 3))

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # 获取当前块
            block = noisy[y:y+block_size, x:x+block_size]
            if block.shape[0] < 3 or block.shape[1] < 3:
                continue  # 跳过过小的块，避免核尺寸问题

            rand = np.random.rand()
            if rand < p_dilate:
                block = cv2.dilate(block, kernel, iterations=1)
            elif rand < p_dilate + p_erode:
                block = cv2.erode(block, kernel, iterations=1)
            # else: 不变

            # 写回到图像
            noisy[y:y+block.shape[0], x:x+block.shape[1]] = block

    return noisy

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

class PosDistSmoother:
    def __init__(self, window_size=5, device="cpu", dtype=torch.float32):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.device = device
        self.dtype = dtype

    def update(self, new_pos_dist_mean):
        """
        添加新的 pos_dist 均值，并返回滑窗中位数（张量）
        """
        if isinstance(new_pos_dist_mean, torch.Tensor):
            new_pos_dist_mean = new_pos_dist_mean.item()
        self.buffer.append(new_pos_dist_mean)
        return self.get_smoothed_median()

    def get_smoothed_median(self):
        """
        返回当前窗口内中位数（torch.Tensor）
        """
        if len(self.buffer) == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        median = statistics.median(self.buffer)
        return torch.tensor(median, device=self.device, dtype=self.dtype)

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

class TripletLoss(nn.Module):
    def __init__(self, in_channels: int = 3, num_embeddings: int = 104):
        super(TripletLoss, self).__init__()
        self.mobilenetv3_like = MobilenetV3_like(in_channels, num_embeddings)

    def forward(self, anchor, positive, negative):
        anchor = self.mobilenetv3_like(anchor)
        positive = self.mobilenetv3_like(positive)
        negative = self.mobilenetv3_like(negative)
        return anchor, positive, negative

class PairDataset(Dataset):
    def __init__(self, root_dir, pairs_per_class=10, use_soft_labels=False, soft_label_range=(0.0, 0.90)):
        self.root_dir = root_dir
        self.use_soft_labels = use_soft_labels  # 是否启用软标签
        self.soft_label_range = soft_label_range  # 软标签随机范围（避免0或1）
        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, img)
                  for img in os.listdir(os.path.join(root_dir, cls))
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        }
        self.classes = list(self.class_to_images.keys())
        self.pairs = self.generate_triplets(pairs_per_class)

    def generate_triplets(self, triplets_per_class):
        triplets = []
        for cls in self.classes:
            images = self.class_to_images[cls]
            if len(images) < 2:
                continue
                
            # 获取其他类的列表（确保有可用的负样本类）
            other_classes = [c for c in self.classes if c != cls and len(self.class_to_images[c]) > 0]
            if not other_classes:
                continue
                
            for _ in range(triplets_per_class):
                # 随机选择anchor和positive（来自同一类）
                anchor, positive = random.sample(images, 2)
                
                # 随机选择一个不同的类作为negative
                neg_cls = random.choice(other_classes)
                negative = random.choice(self.class_to_images[neg_cls])
                
                # 添加到三元组列表
                triplets.append((anchor, positive, negative))
                
        return triplets

    def gray_to_rgb_and_resize(self, img_path):
        block_size = random.randint(12, 24)
        p_dilate = random.uniform(0, 0.15)
        p_erode = random.uniform(0, 0.15)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image {img_path}")
            #设立一张空白图片
            img = np.zeros((224,224),dtype=np.uint8)
            return img
        img = letterbox_gray_cv2(img, target_size=224)
        img = blockwise_morph_noise(img, block_size=block_size, p_dilate=p_dilate, p_erode=p_erode)
        #均值模糊
        img = cv2.blur(img,(3,3))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, positive, negative = self.pairs[idx]
        anchor_img = self.gray_to_rgb_and_resize(anchor)
        positive_img = self.gray_to_rgb_and_resize(positive)
        negative_img = self.gray_to_rgb_and_resize(negative)
        # img1_torch = torch.tensor(img1, dtype=torch.float32)
        # img2_torch = torch.tensor(img2, dtype=torch.float32)
        return (
            transform(anchor_img),
            transform(positive_img),
            transform(negative_img)
        )


class AdaptiveMargin:
    def __init__(self, alpha=0.1, init_pos=0.0, init_neg=1.0):
        self.alpha = alpha  # EMA平滑系数
        self.pos_mean = init_pos  # 正样本距离均值
        self.neg_mean = init_neg  # 负样本距离均值
    
    def ema_margin(self,alhpa=0.1):
        target_margin = (self.neg_mean - self.pos_mean)/(1+self.pos_mean/(self.neg_mean-self.pos_mean))
        margin = (1 - alhpa) * self.get_margin() + alhpa * target_margin
        return margin

    def update(self, pos_dist, neg_dist):
        # 指数移动平均更新
        neg_alhpa = self.alpha / (1 + pos_dist)
        # if pos_dist > self.pos_mean:
        #     self.pos_mean = (1 - neg_alhpa) * self.pos_mean + neg_alhpa * pos_dist
        # else:
        #     self.pos_mean = (1 - self.alpha) * self.pos_mean + self.alpha * pos_dist
        self.pos_mean = (1 - self.alpha) * self.pos_mean + self.alpha * pos_dist
        # if neg_dist > self.neg_mean:
        #     self.neg_mean = (1 - neg_alhpa) * self.neg_mean + neg_alhpa * neg_dist
        # else:
        self.neg_mean = (1 - self.alpha) * self.neg_mean + self.alpha * neg_dist
        return max(self.neg_mean - self.pos_mean,1e-6)  # 返回当前margin
    # def update(self, pos_dist, neg_dist):
    #     margin = neg_dist - pos_dist
    #     self.margin = (1 - self.alpha) * self.margin + self.alpha * margin
    #     return self.margin
    # def update(self, pos_dist, neg_dist):
    #     current_margin = neg_dist - pos_dist

    #     # margin 越大，负样本更新越慢，防止正反馈
    #     neg_alpha = self.alpha / (1 + current_margin)

    #     self.pos_mean = (1 - self.alpha) * self.pos_mean + self.alpha * pos_dist
    #     self.neg_mean = (1 - neg_alpha) * self.neg_mean + neg_alpha * neg_dist

    #     return self.neg_mean - self.pos_mean
    def get_margin(self):
        margin = self.neg_mean - self.pos_mean
        return max(0.5, margin) # 避免margin为负
    def set_alpha_and_init_values(self,alpha=0.1, init_pos=0.0, init_neg=1.0):
        self.alpha = alpha  # EMA平滑系数
        self.pos_mean = init_pos  # 正样本距离均值
        self.neg_mean = init_neg  # 负样本距离均值
        print(f"Margin: alpha={self.alpha}, pos_mean={self.pos_mean}, neg_mean={self.neg_mean}")


class ContrastiveLossAdaptiveMargin(nn.Module):
    def __init__(self, 
                 margin=1.0,
                 device='cpu',
                 winddows_len=10):
        super(ContrastiveLossAdaptiveMargin, self).__init__()
        self.margin = margin
        self.initial_margin = margin
        self.windows = PosDistSmoother(winddows_len,device=device)

        # 状态记录
        self.margin_history = [margin]
        self.val_scores = []
        self.best_val_score = 0.0

    def forward(self, anchor, positive, negative):
        """
        output1, output2: 两个embedding向量 [B, D]
        label: 1表示同类，0表示异类 [B]
        """
        # anchor = F.normalize(anchor, p=2, dim=1)
        # positive = F.normalize(positive, p=2, dim=1)
        # negative = F.normalize(negative, p=2, dim=1)

        distance_pos = F.pairwise_distance(anchor, positive)
        distance_neg = F.pairwise_distance(anchor, negative)
        mean_pos = distance_pos.mean()
        self.windows.update(mean_pos)
        median_pos = self.windows.get_smoothed_median()
        loss = F.relu(distance_pos - distance_neg + self.margin)
        loss2 = F.relu(F.pairwise_distance(median_pos, distance_pos)+self.margin-F.pairwise_distance(median_pos, distance_neg))
        loss3= distance_pos
        loss = loss + 0.5*loss2+0.05*loss3
        return loss.mean(), distance_pos, distance_neg

    def update_adaptive_margin(self,current_margin):
        """根据训练过程调整 margin"""
        self.margin = current_margin
        self.margin_history.append(self.margin)


    def reset(self):
        """重置所有状态"""
        self.margin = self.initial_margin
        self.margin_history = [self.margin]
        self.val_scores = []
        self.best_val_score = 0.0
        self.stagnant_counter = 0

def evaluate_triplet_with_f1(model, val_loader, device, margin=0.0):
    model.eval()
    true_labels = []   # 真实 label: 1 表示 ap 更近（正例）
    pred_labels = []   # 预测 label: 1 表示 d(ap) < d(an)（模型判断正确）

    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a, emb_p, emb_n = model(anchor, positive, negative)
            # #归一化
            # emb_a = F.normalize(emb_a, p=2, dim=1)
            # emb_p = F.normalize(emb_p, p=2, dim=1)
            # emb_n = F.normalize(emb_n, p=2, dim=1)

            d_ap = F.pairwise_distance(emb_a, emb_p)
            d_an = F.pairwise_distance(emb_a, emb_n)

            # 判断模型是否正确分类（满足 d(ap) + margin < d(an)）
            is_correct = (d_an - d_ap > margin).cpu().numpy().astype(int)

            # 三元组真实 label 都是 1（因为 positive 应该比 negative 靠近）
            true_labels.extend([1] * len(is_correct))
            pred_labels.extend(is_correct.tolist())

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"[Triplet Evaluation] Margin={margin:.2f} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    return acc, f1

def find_optimal_threshold(model, val_loader, device, margin_range=None, num_thresholds=20):
    """
    根据验证集距离分布自动选择最大化F1的阈值
    
    参数:
        model: 已加载的模型
        val_loader: 验证集DataLoader（需返回三元组或样本对）
        device: 计算设备（如'cuda'）
        margin_range: 阈值搜索范围 [min, max]，若为None则自动计算
        num_thresholds: 搜索的阈值数量
        
    返回:
        best_threshold: 最优阈值（F1最大）
        best_f1: 对应的F1分数
        all_thresholds: 所有尝试的阈值列表
        f1_scores: 各阈值对应的F1分数
    """
    model.eval()
    distances = []
    labels = []
    
    # 1. 计算所有验证样本对的距离和标签
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, pos, neg = anchor.to(device), positive.to(device), negative.to(device)
            emb_anchor,emb_pos,emb_neg = model(anchor, pos, neg)
            # 计算正负样本对距离
            # 关键修复：处理batch维度距离
            # emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
            # emb_pos = F.normalize(emb_pos, p=2, dim=1)
            # emb_neg = F.normalize(emb_neg, p=2, dim=1)
            pos_dist = F.pairwise_distance(emb_anchor, emb_pos, p=2).cpu().numpy().tolist()
            neg_dist = F.pairwise_distance(emb_anchor, emb_neg, p=2).cpu().numpy().tolist()
            
            distances.extend(pos_dist + neg_dist)
            labels.extend([1]*len(pos_dist) + [0]*len(neg_dist))
                
            # elif len(batch) == 2:  # 样本对数据格式 (img1, img2, label)
            #     img1, img2, label = batch[0].to(device), batch[1].to(device), batch[2]
            #     emb1 = model(img1)
            #     emb2 = model(img2)
                
            #     distances.append(F.pairwise_distance(emb1, emb2, p=2).item())
            #     labels.append(label.item())
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # 2. 确定阈值搜索范围
    if margin_range is None:
        min_thresh = max(0.1, np.percentile(distances[labels == 1], 95))  # 正样本距离的95%分位数
        max_thresh = min(np.percentile(distances[labels == 0], 5), 2.0)    # 负样本距离的5%分位数
        margin_range = [min_thresh, max_thresh]
    
    # 3. 线性空间搜索最优阈值
    all_thresholds = np.linspace(margin_range[0], margin_range[1], num_thresholds)
    f1_scores = []
    
    for thresh in all_thresholds:
        preds = (distances < thresh).astype(int)
        f1 = f1_score(labels, preds)
        f1_scores.append(f1)
    
    # 4. 找到F1最大的阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = all_thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1, all_thresholds, f1_scores

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_soft_cosine_schedule_with_smooth_decay(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=4,
    decay_factor=0.1,  # 最后衰减到的相对幅度
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # 多周期余弦震荡
        cosine_osc = 0.5 * (1 + math.cos(2 * math.pi * num_cycles * progress))
        
        # 平滑包络衰减函数（指数衰减）
        envelope_decay = decay_factor ** progress
        
        # 尾部强制衰减为 0（额外乘一个余弦窗，从 1 到 0）
        end_window = 0.5 * (1 + math.cos(math.pi * progress))  # progress=1 时为0
        
        return max(0.0, envelope_decay * cosine_osc * end_window)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(train_root, test_root, num_epochs=100, patience_limit=10, best_acc=0.0,last_epoch=-1, warmup_epochs=10, num_cycles=2,decay_factor= 0.3):
    
    train_dataset = PairDataset(train_root, pairs_per_class=40,use_soft_labels=True, soft_label_range=(0.0, 0.95))
    test_dataset = PairDataset(test_root, pairs_per_class=40)
    nw = min(os.cpu_count(), 8)
    set_margin = 1
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True,pin_memory=True,num_workers=1,
        persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False,pin_memory=True,num_workers=1,
        persistent_workers=True)
    step_per_epoch = len(train_loader)
    warmup_steps = step_per_epoch * warmup_epochs
    num_epoch_steps = step_per_epoch * num_epochs
    model = TripletLoss().to(device)
    # model.load_state_dict(torch.load(r"model_pth\jus_cnn_se_Triplet_Loss_hard_no_normalize_L2\best_model_best_acc_0.4952_best_f1_0.9489_best_thresh_39.pth", map_location=device))
    criterion = ContrastiveLossAdaptiveMargin(margin=set_margin, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.008)
    scheduler = get_soft_cosine_schedule_with_smooth_decay(optimizer, warmup_steps, num_epoch_steps, num_cycles=num_cycles,decay_factor=decay_factor, last_epoch=last_epoch)

    patience = 0
    best_model_wts = None
    best_f1 = 0.0
    new_margin = 0.0
    best_tr = 0.0
    best_acc = best_acc
    eval_interval = 2
    adaptive_margin = AdaptiveMargin(alpha=0.001)
    for epoch in range(num_epochs):
        pos_dist_list = []
        neg_dist_list = []
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        add_pos_dist = 0.0
        add_neg_dist = 0.0
        # if epoch ==int(warmup_epochs*1.5):
        #     with torch.no_grad():
        #         anchor_img, positive_img, negative_img = next(iter(test_loader))
        #         anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
        #         anchor, positive, negative = model(anchor_img, positive_img, negative_img)
        #         # anchor = F.normalize(anchor, p=2, dim=1)
        #         # positive = F.normalize(positive, p=2, dim=1)
        #         # negative = F.normalize(negative, p=2, dim=1)
        #         pos_dist = F.pairwise_distance(anchor, positive, p=2)
        #         neg_dist = F.pairwise_distance(anchor, negative, p=2)
        #         pos_dist_filtered = remove_outliers(pos_dist.cpu().detach().flatten()).to(pos_dist.device)
        #         neg_dist_filtered = remove_outliers(neg_dist.cpu().detach().flatten()).to(neg_dist.device)
        #         adaptive_margin.set_alpha_and_init_values(alpha=0.03, init_pos=pos_dist_filtered.mean().item(), init_neg=neg_dist_filtered.mean().item())
        for i,(anchor_img, positive_img, negative_img) in enumerate(loop):
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
            # output1,output2 = model(img_a,img_b).squeeze()#元组不能直接squeeze
            anchor,positive,negative = model(anchor_img, positive_img, negative_img)
            # output1 = output1.squeeze()
            # output2 = output2.squeeze()
            anchor = anchor.squeeze()
            positive = positive.squeeze()
            negative = negative.squeeze()
            distance_pos = F.pairwise_distance(anchor, positive)
            distance_neg = F.pairwise_distance(anchor, negative)
            mean_pos = distance_pos.mean().item()
            neg_dist_filtered = remove_outliers(distance_neg.cpu().detach().flatten()).to(distance_neg.device)
            mean_neg = neg_dist_filtered.mean().item()
            adaptive_margin.update(mean_pos, mean_neg)
            current_margin = adaptive_margin.ema_margin()
            criterion.update_adaptive_margin(current_margin)
            loss, pos_dist, neg_dist = criterion(anchor, positive, negative)
            pos_dist_list.append(pos_dist.cpu().detach().numpy())
            neg_dist_list.append(neg_dist.cpu().detach().numpy())
            # # pos_dist_filtered = remove_outliers(pos_dist.cpu().detach().flatten()).to(pos_dist.device)
            # neg_dist_filtered = remove_outliers(neg_dist.cpu().detach().flatten()).to(neg_dist.device)

            # add_pos_dist = pos_dist.mean().item()+add_pos_dist
            # add_neg_dist = neg_dist_filtered.mean().item()+add_neg_dist
            # if (i+1)%15==0 and epoch>=int(warmup_epochs*1.5):
            #     adaptive_margin.update(add_pos_dist/15, add_neg_dist/15)
            #     current_margin = adaptive_margin.ema_margin()
            #     # # print(f"Margin: {current_margin:.4f}")
            #     add_pos_dist = 0.0
            #     add_neg_dist = 0.0
            #     criterion.update_adaptive_margin(current_margin)
            #     current_margin = 0.0
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), margin=criterion.margin,lr=optimizer.param_groups[0]['lr'],mean_pos = adaptive_margin.pos_mean,mean_neg = adaptive_margin.neg_mean)

        avg_loss = running_loss / len(train_loader)
        # acc,f1 = evaluate_triplet_with_f1(model, test_loader,device,criterion.margin*0.7)
        best_tr,f1,_,_ = find_optimal_threshold(model, test_loader,device,margin_range=None, num_thresholds=100)
        plot_pos_neg_dist(pos_dist_list,neg_dist_list,epoch,criterion.margin,best_tr,r"train_dist_image")
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Val Thresh: {best_tr:.4f} - Val F1: {f1:.4f}")
        print(f"Margin: {criterion.margin:.4f}")
        # if (epoch + 1) % eval_interval == 0:
        #     criterion.adjust_margin_from_validation(f1)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model_wts = model.state_dict()
        #     patience = 0
        #     print(f"✅ Best accuracy updated to {best_acc:.4f}")
        if  (epoch + 1) % 5 == 0:
            train_dataset = PairDataset(train_root, pairs_per_class=40,use_soft_labels=True, soft_label_range=(0.0, 0.95))
            train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True,pin_memory=True,num_workers=1,
                persistent_workers=True)
        if  (epoch + 1) % 10 == 0:
            test_dataset = PairDataset(test_root, pairs_per_class=40)
            test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False,pin_memory=True,num_workers=1,
                persistent_workers=True)
        if f1 > best_f1:
            # criterion.adjust_margin_from_validation(f1)
            best_acc = best_tr
            best_epoch = epoch
            best_f1 = f1
            best_model_wts = model.state_dict()
            patience = 0
            print(f"✅ Best F1 score updated to {best_f1:.4f}")
        else:
            if epoch>warmup_epochs*1.5:
                patience += 1
                print(f"⏳ Patience {patience}/{patience_limit}")
            if patience >= patience_limit:
                print(f"⛔ Early stopping.best_tr: {best_acc:.4f}, best_f1: {best_f1:.4f}, best_thresh: {best_epoch}")
                break

    # 保存最优模型
    print(f"Best Accuracy: {best_acc:.4f}, Best F1: {best_f1:.4f}, Best Thresh: {best_epoch}")
    os.makedirs("model_pth/jus_cnn_se_Triplet_Loss_hard_no_normalize_L2", exist_ok=True)
    torch.save(best_model_wts, f"model_pth/jus_cnn_se_Triplet_Loss_hard_no_normalize_L2/best_model_best_acc_{best_acc:.4f}_best_f1_{best_f1:.4f}_best_thresh_{best_epoch}.pth")
    duration = 1500  # 毫秒
    freq = 440  # 频率 (Hz)
    winsound.Beep(freq, duration)
    plt.plot(criterion.margin_history)
    plt.xlabel("Evaluation Epoch")
    plt.ylabel("Margin")
    plt.title("Margin Evolution")
    plt.grid(True)
    plt.show()

# ======= 启动训练 =======
if __name__ == "__main__":
    train(
        train_root=r"date\shape_date\can_use\train",
        test_root=r"date\shape_date\can_use\test",
        num_epochs=150,
        patience_limit=50,
        num_cycles=4,
        last_epoch=-1,
        warmup_epochs=3,
        decay_factor = 0.3
    )