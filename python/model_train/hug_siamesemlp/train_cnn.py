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

# ======= CNN-based Feature Extractor =======
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Adaptive pooling to ensure fixed output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate the size after adaptive pooling
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ======= 三元组数据集类 (Updated for CNN) =======
class CNNTripletDataset(Dataset):
    def __init__(self, root_dir, triplets_per_class=10, img_size=(128, 128)):
        self.root_dir = root_dir
        self.img_size = img_size
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
        def load_image(path):
            img = Image.open(path).convert('L')  # Convert to grayscale
            img = img.resize(self.img_size)
            img = np.array(img, dtype=np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize to [0, 1]
            img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
            return img

        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor = load_image(anchor_path)
        positive = load_image(pos_path)
        negative = load_image(neg_path)
        return anchor, positive, negative

# ======= CNN-based Triplet Network =======
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.cnn = CNNFeatureExtractor()
        
    def forward_once(self, x):
        return self.cnn(x)
    
    def forward(self, anchor, positive, negative):
        return self.forward_once(anchor), self.forward_once(positive), self.forward_once(negative)

# ======= 评估函数 =======
def evaluate(model, dataloader, margin=1.0):
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

    train_dataset = CNNTripletDataset(train_root, triplets_per_class=100, img_size=(128, 128))
    test_dataset = CNNTripletDataset(test_root, triplets_per_class=15, img_size=(128, 128))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TripletNet().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate for CNN
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
    os.makedirs("model_pth/tripletcnn", exist_ok=True)
    torch.save(best_model_wts, "model_pth/tripletcnn/triplet_cnn_trained.pth")

# ======= 启动训练 =======
if __name__ == "__main__":
    train(
        train_root=r"date\MPEG7_dataset\train",
        test_root=r"date\MPEG7_dataset\val",
        num_epochs=200,
        Patience=20,
        num_cycles=0.8
    )