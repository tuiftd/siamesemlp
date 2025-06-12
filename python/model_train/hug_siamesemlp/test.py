import os
import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

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
    def __init__(self):
        super(TripletNet, self).__init__()
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


def evaluate_metrics(model, dataloader, margin=1):
    model.eval()
    y_true = []
    y_pred = []
    distances = []
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc='Evaluating Metrics'):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # 获取模型输出
            anc_out, pos_out, neg_out = model(anchor, positive, negative)
            
            # 计算距离
            pos_dist = F.pairwise_distance(anc_out, pos_out)
            neg_dist = F.pairwise_distance(anc_out, neg_out)
            
            # 预测结果 (1表示正样本对正确分类，0表示负样本对正确分类)
            preds = (pos_dist + margin < neg_dist).cpu().numpy()
            
            # 真实标签 (1表示正样本对，0表示负样本对)
            # 我们交替处理正样本对和负样本对
            batch_size = anchor.size(0)
            true_labels = np.array([1]*batch_size + [0]*batch_size)
            
            # 对于正样本对，预测距离应该小
            # 对于负样本对，预测距离应该大
            y_true.extend(true_labels)
            y_pred.extend(np.concatenate([preds, ~preds]))
            
            # 收集距离用于分析
            distances.extend(pos_dist.cpu().numpy())
            distances.extend(neg_dist.cpu().numpy())
    
    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative Pairs', 'Positive Pairs']))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative Pairs', 'Positive Pairs'],
                yticklabels=['Negative Pairs', 'Positive Pairs'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 绘制距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(distances[:len(distances)//2], bins=50, alpha=0.5, label='Positive Pairs Distance')
    plt.hist(distances[len(distances)//2:], bins=50, alpha=0.5, label='Negative Pairs Distance')
    plt.axvline(x=margin, color='r', linestyle='--', label='Margin')
    plt.title('Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances)
    }

if __name__ == "__main__":
    # 加载模型
    model = TripletNet().to(device)
    model.load_state_dict(torch.load(r"model_pth\tripletmlp\triplet_action_hu_trained_1.pth"))
    
    # 创建数据集和数据加载器
    test_root = r"date\MPEG7_dataset\val"
    test_dataset = HuTripletDataset(test_root, triplets_per_class=64)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估模型
    metrics = evaluate_metrics(model, test_loader)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Distance Mean: {metrics['distance_mean']:.4f}")
    print(f"Distance Std: {metrics['distance_std']:.4f}")