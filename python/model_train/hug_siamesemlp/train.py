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
                  if img.lower().endswith(('.png', '.jpg', '.jpeg','.gif'))]
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
                pairs.append((img1, img3, -1))  # 负样本
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
        self.dropout1 = nn.Dropout(p=0.2)

    def forward_once(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ======= 评估函数 =======
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, label in tqdm(dataloader, desc='Evaluating', leave=False):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out1, out2 = model(x1, x2)
            sim = F.cosine_similarity(out1, out2)
            preds = torch.where(sim > 0.5, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
            correct += torch.sum(preds == label).item()
            total += label.size(0)
    return correct / total

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Args:
        optimizer: 优化器对象（如Adam）
        num_warmup_steps: Warmup步数（线性增长阶段）
        num_training_steps: 总训练步数（包括warmup）
        num_cycles: 余弦周期数（默认0.5表示半周期衰减）
        last_epoch: 当前epoch（用于恢复训练）
    """
    def lr_lambda(current_step: int):
        # Warmup阶段：线性增长学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ======= 主训练函数 =======
def train(train_root, test_root, num_epochs=20,Patience=5,best_acc=0.0,last_epoch=-1,warmup_steps=10,num_cycles=0.5):
    best_acc = 0.0
    patience_count = 0
    best_model_wts = None
    train_dataset = HuMomentPairDataset(train_root, pairs_per_class=100)
    test_dataset  = HuMomentPairDataset(test_root,  pairs_per_class=15)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    model = SiameseNet().to(device)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs,last_epoch=last_epoch,num_cycles=num_cycles)

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
        acc = evaluate(model, test_loader)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Test Accuracy: {acc:.4f}\n")
        if acc > best_acc:
            best_acc = acc
            patience_count = 0
            best_model_wts = model.state_dict()
            print(f"Best accuracy updated to {best_acc:.4f}")
        else:
            if epoch > warmup_steps:
                patience_count += 1
            print(f"Patience count: {patience_count}/{Patience}")
            if patience_count >= Patience:
                print(f"Early stopping at epoch {epoch+1} with best accuracy {best_acc:.4f}")
                break


    torch.save(best_model_wts, r"model_pth\siamesemlp\siamese_hu_trained.pth")

# ======= 启动训练 =======
if __name__ == "__main__":
    train(train_root=r"date\MPEG7_dataset\train", test_root=r"date\MPEG7_dataset\val", num_epochs=200, Patience=20,num_cycles=2)  # ← 请替换为你的数据路径
