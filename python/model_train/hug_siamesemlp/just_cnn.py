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

gc.collect()
torch.cuda.empty_cache() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
transform = transforms.ToTensor()

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
        self.pairs = self.generate_pairs(pairs_per_class)

    def generate_pairs(self, pairs_per_class):
        pairs = []
        for cls in self.classes:
            images = self.class_to_images[cls]
            if len(images) < 2:
                continue
            # for _ in range(pairs_per_class):
            #     img1, img2 = random.sample(images, 2)
            #     pairs.append((img1, img2, 1))  # 正样本
            #     neg_cls = random.choice([c for c in self.classes if c != cls and len(self.class_to_images[c]) > 0])
            #     img3 = random.choice(self.class_to_images[neg_cls])
            #     pairs.append((img1, img3, 0))  # 负样本
            # 正样本对（同类图像）
            for _ in range(pairs_per_class):
                a, b = random.sample(images, 2)
                label = 1.0  # 硬标签
                if self.use_soft_labels:
                    # 在接近1的范围内随机（如 [0.8, 0.95]）
                    label = random.uniform(self.soft_label_range[1] - 0.05, self.soft_label_range[1])
                    if label>1.0:
                        label = 1.0
                pairs.append((a, b, label))

            # 负样本对（不同类图像）
            for _ in range(pairs_per_class):
                other_cls = random.choice([c for c in self.classes if c != cls and len(self.class_to_images[c]) > 0])
                a = random.choice(images)
                b = random.choice(self.class_to_images[other_cls])
                label = 0.0  # 硬标签
                # if self.use_soft_labels:
                #     # 在接近0的范围内随机（如 [0.05, 0.2]）
                #     label = random.uniform(self.soft_label_range[0], self.soft_label_range[0] + 0.1)
                pairs.append((a, b, label))
        return pairs

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
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self.gray_to_rgb_and_resize(img1_path)
        img2 = self.gray_to_rgb_and_resize(img2_path)
        # img1_torch = torch.tensor(img1, dtype=torch.float32)
        # img2_torch = torch.tensor(img2, dtype=torch.float32)
        return (
            transform(img1),
            transform(img2),
            torch.tensor(label, dtype=torch.float32)
        )


class ContrastiveLoss(nn.Module):#直接计算欧几里得距离
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 欧几里得距离
        distance = F.pairwise_distance(output1, output2)
        # 对比损失公式
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()

class ContrastiveLossNormalized(nn.Module):# L2 归一化后再计算距离
    def __init__(self, margin=1.0):
        super(ContrastiveLossNormalized, self).__init__()
        self.margin = margin

    def set_margin(self, new_margin):
        self.margin = new_margin

    def forward(self, output1, output2, label):
        # L2 归一化后再计算距离
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)
        distance = F.pairwise_distance(output1, output2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()

class ContrastiveLossMahalanobis(nn.Module):# 马氏距离版本
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, margin=1.0):
        super(ContrastiveLossMahalanobis, self).__init__()
        self.margin = margin
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std + 1e-8, requires_grad=False)

    def forward(self, output1, output2, label):
        # 标准化（z-score）
        output1_norm = (output1 - self.mean) / self.std
        output2_norm = (output2 - self.mean) / self.std
        distance = F.pairwise_distance(output1_norm, output2_norm)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()
    # 假设模型为 SiameseNet
    # all_embeddings = []

    # model.eval()
    # with torch.no_grad():
    #     for x1, x2, _ in train_loader:
    #         emb1, emb2 = model(x1.to(device), x2.to(device))
    #         all_embeddings.append(emb1)
    #         all_embeddings.append(emb2)

    # all_embeddings = torch.cat(all_embeddings, dim=0)
    # mean = all_embeddings.mean(dim=0)
    # std = all_embeddings.std(dim=0) + 1e-8

    # # 用于创建马氏损失
    # mahalanobis_loss = ContrastiveLossMahalanobis(mean, std)
class ContrastiveLossAdaptiveMargin(nn.Module):
    def __init__(self, 
                 margin=1.0,
                 min_margin=0.5,
                 max_margin=2.0,
                 decay_factor=0.95,
                 growth_factor=1.05,
                 stagnant_threshold=3,
                 try_down_threshold=3,
                 drop_threshold=0.05):
        super(ContrastiveLossAdaptiveMargin, self).__init__()
        self.margin = margin
        self.initial_margin = margin

        # Margin控制参数
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.decay_factor = decay_factor
        self.growth_factor = growth_factor

        # 验证反馈控制
        self.stagnant_threshold = stagnant_threshold  # 多轮无提升判定阈值
        self.drop_threshold = drop_threshold          # 单次性能暴跌判断阈值
        self.try_down_threshold = try_down_threshold    # 连续降低判定阈值

        # 状态记录
        self.margin_history = [margin]
        self.val_scores = []
        self.best_val_score = 0.0
        self.stagnant_counter = 0
        self.try_down_counter = 0

    def forward(self, output1, output2, label):
        """
        output1, output2: 两个embedding向量 [B, D]
        label: 1表示同类，0表示异类 [B]
        """
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        distance = F.pairwise_distance(output1, output2)

        loss_same = label * distance.pow(2)
        loss_diff = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = loss_same + loss_diff
        return loss.mean()

    def decay_margin(self):
        """训练过程中逐步减小 margin"""
        new_margin = max(self.margin * self.decay_factor, self.min_margin)
        if new_margin != self.margin:
            print(f"[Margin Decay] Margin {self.margin:.4f} → {new_margin:.4f}")
            self.margin = new_margin
            self.margin_history.append(self.margin)

    def adjust_margin_from_validation(self, val_score):
        """根据验证集表现调整 margin"""
        self.val_scores.append(val_score)

        # 情况1：性能显著下降 → 缩小 margin（更容易区分）
        if val_score < self.best_val_score - self.drop_threshold and val_score > self.best_val_score - 2*self.drop_threshold:
            new_margin = max(self.margin * self.decay_factor, self.min_margin)
            print(f"[Val Drop] Margin {self.margin:.4f} → {new_margin:.4f} due to F1 drop")
            self.margin = new_margin
            self.try_down_counter = 0
        elif self.try_down_counter >= self.try_down_threshold:
            new_margin = max(self.margin * self.decay_factor, self.min_margin)
            print(f"[Val Drop] Margin {self.margin:.4f} → {new_margin:.4f} due to continuous F1 stagnation")
            self.margin = new_margin
            self.stagnant_counter = 0
        # 情况2：连续多轮停滞 → 增加 margin（加大学习难度）
        elif val_score < self.best_val_score:
            self.stagnant_counter += 1
            if self.stagnant_counter >= self.stagnant_threshold:
                new_margin = min(self.margin * self.growth_factor, self.max_margin)
                print(f"[Val Stagnant] Margin {self.margin:.4f} → {new_margin:.4f} after {self.stagnant_counter} stagnant rounds")
                self.margin = new_margin
                self.stagnant_counter = 0
                self.try_down_counter += 1

        # 情况3：性能提升
        elif val_score > self.best_val_score:
            self.best_val_score = val_score
            self.stagnant_counter = 0
            self.try_down_counter = 0

        self.margin_history.append(self.margin)

    def reset(self):
        """重置所有状态"""
        self.margin = self.initial_margin
        self.margin_history = [self.margin]
        self.val_scores = []
        self.best_val_score = 0.0
        self.stagnant_counter = 0

def collect_distances_labels(model, val_loader, device):
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, labels in val_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            emb1, emb2 = model(x1, x2)

            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            distances = F.pairwise_distance(emb1, emb2)  # [B]
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_distances), np.array(all_labels)

def find_best_threshold(distances, labels, metric="f1"):
    thresholds = np.linspace(0.2, 2.0, 200)
    best_score = 0
    best_thresh = 0

    for t in thresholds:
        preds = (distances < t).astype(int)
        score = f1_score(labels, preds) if metric == "f1" else accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score

def evaluate_with_auto_threshold(model, val_loader, device, metric="f1"):
    distances, labels = collect_distances_labels(model, val_loader, device)
    best_thresh, best_score = find_best_threshold(distances, labels, metric=metric)

    preds = (distances < best_thresh).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"[Auto Evaluation] Best Threshold: {best_thresh:.4f} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    return best_thresh, acc, f1

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(train_root, test_root, num_epochs=100, patience_limit=10, best_acc=0.0,last_epoch=-1, warmup_steps=10, num_cycles=2):
    train_dataset = PairDataset(train_root, pairs_per_class=60,use_soft_labels=True, soft_label_range=(0.0, 0.95))
    test_dataset = PairDataset(test_root, pairs_per_class=60)
    nw = min(os.cpu_count(), 8)
    set_margin = 2.0
    train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True,pin_memory=True,num_workers=1,
        persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False,pin_memory=True,num_workers=1,
        persistent_workers=True)

    model = SiameseNet().to(device)
    criterion = ContrastiveLossAdaptiveMargin(margin=set_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.008)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs, num_cycles=num_cycles, last_epoch=last_epoch)

    patience = 0
    best_model_wts = None
    best_f1 = 0.0
    new_margin = 0.0
    best_tr = 0.0
    best_acc = best_acc
    eval_interval = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for img_a,img_b,label in loop:
            img_a,img_b,label = img_a.to(device), img_b.to(device), label.to(device)
            # output1,output2 = model(img_a,img_b).squeeze()#元组不能直接squeeze
            output1,output2 = model(img_a,img_b)
            output1 = output1.squeeze()
            output2 = output2.squeeze()
            loss = criterion(output1, output2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        best_thresh,acc,f1 = evaluate_with_auto_threshold(model, test_loader,device)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Val Acc: {acc:.4f} - Val F1: {f1:.4f}- Best Thresh: {best_thresh:.4f}")
        if (epoch + 1) % eval_interval == 0:
            criterion.adjust_margin_from_validation(f1)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model_wts = model.state_dict()
        #     patience = 0
        #     print(f"✅ Best accuracy updated to {best_acc:.4f}")
        if  (epoch + 1) % 5 == 0:
            train_dataset = PairDataset(train_root, pairs_per_class=60,use_soft_labels=True, soft_label_range=(0.0, 0.95))
            train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True,pin_memory=True,num_workers=1,
                persistent_workers=True)
        if  (epoch + 1) % 30 == 0:
            test_dataset = PairDataset(test_root, pairs_per_class=60)
            test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False,pin_memory=True,num_workers=1,
                persistent_workers=True)
        if f1 > best_f1:
            criterion.adjust_margin_from_validation(f1)
            best_acc = acc
            best_tr = best_thresh
            best_f1 = f1
            best_model_wts = model.state_dict()
            patience = 0
            print(f"✅ Best F1 score updated to {best_f1:.4f}")
        else:
            if epoch>warmup_steps*1.5:
                patience += 1
                print(f"⏳ Patience {patience}/{patience_limit}")
            if patience >= patience_limit:
                print(f"⛔ Early stopping.best_acc: {best_acc:.4f}, best_f1: {best_f1:.4f}, best_thresh: {best_tr:.4f}")
                break

    # 保存最优模型
    print(f"Best Accuracy: {best_acc:.4f}, Best F1: {best_f1:.4f}, Best Thresh: {best_tr:.4f}")
    os.makedirs("model_pth/jus_cnn_se", exist_ok=True)
    torch.save(best_model_wts, f"model_pth/jus_cnn_se/best_model_best_acc_{best_acc:.4f}_best_f1_{best_f1:.4f}_best_thresh_{best_tr:.4f}.pth")
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
        num_epochs=100,
        patience_limit=20,
        num_cycles=2.3,
        last_epoch= -1,
        warmup_steps=8
    )