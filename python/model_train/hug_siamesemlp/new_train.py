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
from torch.optim.lr_scheduler import LambdaLR
import winsound
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
def letterbox_gray(image, target_size=256):
    # 原始图像尺寸 (注意PIL是width, height顺序)
    w, h = image.size
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    # 等比例缩放后的新尺寸
    new_w, new_h = int(w * scale), int(h * scale)
    # 缩放图像 (使用PIL的LANCZOS高质量重采样)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    canvas = Image.new('L', (target_size, target_size), (0))
    
    # 将缩放后的图像居中放置
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))
    
    return canvas
# ======= Hu 矩特征提取 =======
def compute_log_hu_moments_and_canny(image_path):
    img = Image.open(image_path).convert('L')
    img = letterbox_gray(img, target_size=256)
    # img = transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.0))(img)
    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_np)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    canny = cv2.Canny(mask, 100, 200)
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m)
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu.flatten(), canny,mask

# ======= Hu 特征相似性数据集（图像对 + 标签）=======
class HuPairDataset_and_canny(Dataset):
    def __init__(self, root_dir, pairs_per_class=20, use_soft_labels=False, soft_label_range=(0.0, 0.80)):
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
            
            # 正样本对（同类图像）
            for _ in range(pairs_per_class):
                a, b = random.sample(images, 2)
                label = 1.0  # 硬标签
                if self.use_soft_labels:
                    # 在接近1的范围内随机（如 [0.8, 0.95]）
                    label = random.uniform(self.soft_label_range[1] - 0.1, self.soft_label_range[1])
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path, label = self.pairs[idx]
        a ,canny_a,mask_a = compute_log_hu_moments_and_canny(a_path)
        b ,canny_b,mask_b = compute_log_hu_moments_and_canny(b_path)
        img_a_torch = torch.cat([torch.tensor(canny_a, dtype=torch.float32).unsqueeze(0),torch.tensor(mask_a, dtype=torch.float32).unsqueeze(0)],dim=0)
        img_b_torch = torch.cat([torch.tensor(canny_b, dtype=torch.float32).unsqueeze(0),torch.tensor(mask_b, dtype=torch.float32).unsqueeze(0)],dim=0)
        return (
            torch.tensor(a, dtype=torch.float32),
            img_a_torch,
            torch.tensor(b, dtype=torch.float32),
            img_b_torch,
            torch.tensor(label, dtype=torch.float32)
        )
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels, stride=1,SE=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.pointwise1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=mid_channels, bias=False)
        self.pointwise2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.SE = SqueezeExcite(mid_channels) if SE else nn.Identity()
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.SE(x)
        x = self.pointwise2(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcite, self).__init__()
        reduced_channels = in_channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True)  # 或使用 nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)       # [B, C, 1, 1]
        scale = self.se(scale)     # [B, C, 1, 1]
        return x * scale           # 逐通道缩放

# ======= 模型结构 =======
class HuNetAttn_1D(nn.Module):
    def __init__(self, channel=7, reduction=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.BatchNorm1d(channel),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(2,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(40,80),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(80,16))
        self.last_Linear = nn.Linear(2, 16)
        self.bn_last = nn.BatchNorm1d(16)
        self.dropout_last = nn.Dropout(p=0.2)
        self.head = nn.Linear(16, 1)
        self.DW1 = DepthwiseSeparableConv(16, 64, 24, 2)
        self.DW2 = DepthwiseSeparableConv(24, 72, 24, 1, SE=True)
        #最大池化
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DW3 = DepthwiseSeparableConv(24, 72, 40, 2)
        self.DW4 = DepthwiseSeparableConv(40, 120, 40, 1, SE=True)
        self.Pointwise1 = nn.Sequential(
        nn.Conv2d(16, 24, kernel_size=1, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(inplace=True),
        )
        self.Pointwise3 =nn.Sequential(
        nn.Conv2d(24, 40, kernel_size=1),
        nn.BatchNorm2d(40),
        nn.ReLU(inplace=True),
        )


        # self.DW5 = DepthwiseSeparableConv(40, 120, 40, 1, SE=True)
    def canny_conv(self,img):
        x = self.conv1(img)
        out = self.DW1(x)
        out = out+self.Maxpool(self.Pointwise1(x))
        out = self.DW2(out) + out
        out = self.DW3(out)+self.Maxpool(self.Pointwise3(out))
        out = self.DW4(out) + out
        # out = self.DW5(out) + out
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return F.normalize(out, p=2, dim=1)



    def forward(self, hu_a, img_a, hu_b, img_b):
        D_16_a = self.canny_conv(img_a)
        D_16_b = self.canny_conv(img_b)
        D_1 = F.cosine_similarity(D_16_a, D_16_b, dim=1)
        D_1 = D_1.unsqueeze(1)
        w_a = self.attn(hu_a)
        w_b = self.attn(hu_b)
        hu_a = hu_a + hu_a * w_a
        hu_b = hu_b + hu_b * w_b
        d_hu = torch.sum(torch.abs(hu_b - hu_a), 1, keepdim=True)
        features = torch.cat([D_1, d_hu], dim=1)
        features = self.last_Linear(features)
        features = self.bn_last(features)
        features = F.relu(features)
        features = self.dropout_last(features)
        return torch.sigmoid(self.head(features))

# ======= 评估函数 =======
def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for a, img_a,b,img_b,label in tqdm(dataloader, desc="Evaluating", leave=False):
            a, img_a,b,img_b,label = a.to(device), img_a.to(device), b.to(device), img_b.to(device), label.to(device)
            output = model(a,img_a,b,img_b)
            preds = (output >= threshold).float()
            correct += (preds.squeeze() == label).sum().item()
            total += label.size(0)
    return correct / total
#=======优化器=======
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)
# ======= 训练入口 =======

def train(train_root, test_root, num_epochs=100, patience_limit=10, best_acc=0.0,last_epoch=-1, warmup_steps=10, num_cycles=2):
    train_dataset = HuPairDataset_and_canny(train_root, pairs_per_class=80,use_soft_labels=True, soft_label_range=(0.0, 0.9))
    test_dataset = HuPairDataset_and_canny(test_root, pairs_per_class=30)
    nw = min(os.cpu_count(), 8)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,pin_memory=True,num_workers=1,
        persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,pin_memory=True,num_workers=1,
        persistent_workers=True)

    model = HuNetAttn_1D().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs, num_cycles=num_cycles)

    patience = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for a, img_a,b,img_b,label in loop:
            # a, b, canny_a, canny_b, mask_a,mask_b,label = a.to(device), b.to(device), canny_a.to(device), canny_b.to(device),mask_a.to(device),mask_b.to(device), label.to(device)
            a, img_a,b,img_b,label = a.to(device), img_a.to(device), b.to(device), img_b.to(device), label.to(device)
            output = model(a,img_a,b,img_b).squeeze()
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = model.state_dict()
            patience = 0
            print(f"✅ Best accuracy updated to {best_acc:.4f}")
        else:
            if epoch>warmup_steps*1.5:
                patience += 1
                print(f"⏳ Patience {patience}/{patience_limit}")
            if patience >= patience_limit:
                print("⛔ Early stopping.")
                break

    # 保存最优模型
    os.makedirs("model_pth/huattn_dwcv_se", exist_ok=True)
    torch.save(best_model_wts, "model_pth/huattn_dwcv_se/hu_bn_attn_dwcv_se_net_best1.pth")
    duration = 1500  # 毫秒
    freq = 440  # 频率 (Hz)
    winsound.Beep(freq, duration)

# ======= 启动训练 =======
if __name__ == "__main__":
    train(
        train_root=r"dataset\train",
        test_root=r"dataset\test",
        num_epochs=100,
        patience_limit=20,
        num_cycles=2,
        last_epoch= -1,
        warmup_steps=10
    )
