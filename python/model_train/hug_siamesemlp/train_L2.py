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

# ======= 设置GPU/CPU =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ======= Hu矩特征提取 =======
def compute_log_hu_moments(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(img_np)
    hu = cv2.HuMoments(m)
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu.flatten()

# ======= 数据集类：加载成对图像和标签 =======
class HuMomentPairDataset(Dataset):
    def __init__(self, root_dir, pairs_per_class=10):
        self.root_dir = root_dir
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
            for _ in range(pairs_per_class):
                img1, img2 = random.sample(images, 2)
                pairs.append((img1, img2, 1))  # 正样本
                neg_cls = random.choice([c for c in self.classes if c != cls and len(self.class_to_images[c]) > 0])
                img3 = random.choice(self.class_to_images[neg_cls])
                pairs.append((img1, img3, 0))  # 负样本
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        hu1 = compute_log_hu_moments(img1_path)
        hu2 = compute_log_hu_moments(img2_path)
        return (
            torch.tensor(hu1, dtype=torch.float32),
            torch.tensor(hu2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# ======= 网络结构 =======
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout1 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(32)

    def forward_once(self, x):
        x_hu = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        x = torch.cat((x_hu, x), dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ======= 对比损失函数（欧氏距离） =======
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss)

# ======= 学习率调度器 =======
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# ======= 评估函数：使用欧氏距离判断是否相似 =======
def evaluate(model, dataloader, threshold=0.7):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, label in tqdm(dataloader, desc='Evaluating', leave=False):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out1, out2 = model(x1, x2)
            dist = F.pairwise_distance(out1, out2)
            preds = torch.where(dist < threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            correct += torch.sum(preds == label).item()
            total += label.size(0)
    return correct / total

# ======= 主训练函数 =======
def train(train_root, test_root, num_epochs=20, patience=5, best_acc=0.0, warmup_steps=10, num_cycles=0.5):
    train_dataset = HuMomentPairDataset(train_root, pairs_per_class=100)
    test_dataset = HuMomentPairDataset(test_root, pairs_per_class=15)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SiameseNet().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs, num_cycles)

    best_model_wts = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x1, x2, label in loop:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, threshold=0.5)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = model.state_dict()
            patience_counter = 0
            print(f"New best accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping")
                break

    if best_model_wts:
        model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), r"model_pth\siamesemlp\contrastive_siamese_hu_cut.pth")

# ======= 启动训练 =======
if __name__ == "__main__":
    train(train_root=r"date\MPEG7_dataset\train", test_root=r"date\MPEG7_dataset\val",
          num_epochs=100, patience=15, warmup_steps=5, num_cycles=2)
