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

# ======= 三元组数据集类 =======
class HuTripletDataset(Dataset):
    def __init__(self, root_dir, triplets_per_class=10):
        self.root_dir = root_dir
        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, img)
                  for img in os.listdir(os.path.join(root_dir, cls))
                  if img.lower().endswith(('.png', '.jpg', '.jpeg','.gif'))]
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        }
        self.classes = list(self.class_to_images.keys())
        self.triplets = self.generate_triplets(triplets_per_class)

    def generate_triplets(self, triplets_per_class):
        triplets = []
        for cls in self.classes:
            images = self.class_to_images[cls]
            if len(images) < 2:
                continue
            for _ in range(triplets_per_class):
                anchor, positive = random.sample(images, 2)
                neg_cls = random.choice([c for c in self.classes if c != cls and len(self.class_to_images[c]) > 0])
                negative = random.choice(self.class_to_images[neg_cls])
                triplets.append((anchor, positive, negative))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor = compute_log_hu_moments(anchor_path)
        positive = compute_log_hu_moments(pos_path)
        negative = compute_log_hu_moments(neg_path)
        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32)
        )

# ======= MLP 网络结构 =======
class TripletNet(nn.Module):
    def __init__(self, channel=7, reduction=4):
        super(TripletNet, self).__init__()
        self.attn = nn.Sequential(
            nn.BatchNorm1d(channel),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channel // reduction),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.bn0 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 64, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1, bias=False)
        self.head = nn.Linear(2, 1)
        # self.bn3 = nn.BatchNorm1d(23)

    def _embed16(self, x_raw):
        x= x_raw
        x = self.bn0(x)
        x= self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        return self.fc2(x)
    
    def forward_once(self, hu):
        w = self.attn(hu)
        hu = hu * w+ hu
        return torch.cat([
            torch.sum(torch.abs(hu - hu.mean(0, keepdim=True)), 1, keepdim=True),  # 1D: 与批均 Hu 差，可改为恒 0
            self._embed16(hu)
        ], 1) 

    def forward(self, anchor, positive, negative):
        f_anchor   = self.forward_once(anchor)
        f_positive = self.forward_once(positive)
        f_negative = self.forward_once(negative)
        return f_anchor, f_positive, f_negative

# ======= 评估函数 =======
def evaluate(model, dataloader, margin=0.8):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc='Evaluating', leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anc_out, pos_out, neg_out = model(anchor, positive, negative)
            pos_dist = F.pairwise_distance(anc_out, pos_out)
            neg_dist = F.pairwise_distance(anc_out, neg_out)
            correct += torch.sum(pos_dist + margin < neg_dist).item()
            total += anchor.size(0)
    return correct / total

# ======= Cosine Scheduler with Warmup =======
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# ======= 主训练函数 =======
def train(train_root, test_root, num_epochs=100, Patience=10, best_acc=0.0, last_epoch=-1, warmup_steps=10, num_cycles=2):
    best_model_wts = None
    patience_count = 0
   
    train_dataset = HuTripletDataset(train_root, triplets_per_class=100)
    test_dataset = HuTripletDataset(test_root, triplets_per_class=15)
    nw = min(os.cpu_count(), 32 if 32 > 1 else 0, 8)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=nw,pin_memory=True,
        persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=nw,pin_memory=True,
        persistent_workers=True)

    model = TripletNet().to(device)
    # model.load_state_dict(torch.load("model_pth/tripletmlp/triplet_hu_trained_3.pth"))
    criterion = nn.TripletMarginLoss(margin=0.8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs, num_cycles=num_cycles)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for anchor, positive, negative in loop:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anc_out, pos_out, neg_out = model(anchor, positive, negative)
            loss = criterion(anc_out, pos_out, neg_out)

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
            best_model_wts = model.state_dict()
            patience_count = 0
            print(f"Best accuracy updated to {best_acc:.4f}")
        else:
            if epoch > warmup_steps:
                patience_count += 1
                print(f"Patience: {patience_count}/{Patience}")
                if patience_count >= Patience:
                    print(f"Early stopping at epoch {epoch+1}, best accuracy: {best_acc:.4f}")
                    break

    # 保存最优模型
    os.makedirs("model_pth/tripletmlp", exist_ok=True)
    torch.save(best_model_wts, "model_pth/tripletmlp/triplet_action_hu_trained_1.pth")

# ======= 启动训练 =======
if __name__ == "__main__":
    train(
        train_root=r"date\MPEG7_dataset\train",
        test_root=r"date\MPEG7_dataset\val",
        num_epochs=130,
        Patience=20,
        num_cycles=1.5,
        last_epoch= -1,
    )
