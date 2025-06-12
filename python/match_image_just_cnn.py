# match_with_cached_templates_attn.py
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
from new_new_new_shape_lcp import shape_context_matching ,radian_to_degree
import time
transform = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def padding_to_size(img, size):
    """
    padding image to size
    """
    h, w = img.shape[:2]
    top, bottom, left, right = 0, 0, 0, 0
    if h < size:
        dh = size - h
        top = dh // 2
        bottom = dh - top
    elif h > size:
        top = (h - size) // 2
        bottom = h - size - top
    if w < size:
        dw = size - w
        left = dw // 2
        right = dw - left
    elif w > size:
        left = (w - size) // 2
        right = w - size - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return img
def letterbox_gray_cv2(image, target_size=256):
    """输入图像为灰度图，返回等比例缩放后的三通道灰度图"""
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
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return canvas

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
class TripletNet_2_branch(nn.Module):
    def __init__(self, net_obj):
        super(TripletNet_2_branch, self).__init__()
        self.net_obj = net_obj
    def forward(self, x1, x2):
        x1 = self.net_obj(x1)
        x2 = self.net_obj(x2)
        return x1, x2
# ========= 设备 =========


# ========= Hu 矩 =========
# def compute_log_hu_moments(img=None, image_path=None):
#     if img is None:
#         img = Image.open(image_path).convert('L')
#         img = letterbox_gray(img, target_size=256)
#         img = np.array(img)
#     img = letterbox_gray(img, target_size=256)
#     img = np.array(img)
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros_like(binary)
#     cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
#     canny = cv2.Canny(mask, 100, 200)
#     # cv2.imshow("contour_mask", mask)
#     # cv2.waitKey(0)
#     m = cv2.moments(mask)
#     hu = cv2.HuMoments(m)
#     log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
#     return log_hu.flatten(),canny,mask
def edge_to_binary(canny_img):
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(canny_img)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        cv2.drawContours(mask, [cnt], 0, 255, cv2.FILLED)
    mask = letterbox_gray_cv2(mask, target_size=int(224*0.8))
    # cv2.imshow("contour_mask", mask)
    # cv2.waitKey(0)

    mask = padding_to_size(mask, 224)
    mask = cv2.blur(mask,(3,3))
    # cv2.imshow("contour_mask", mask)
    # cv2.waitKey(0)
    #均值模糊

    # cv2.imshow("contour_mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mask

def score_match(distance, thr):
    print(distance,thr)
    if thr == 0:
        raise ValueError("thr should not be 0")
    if distance < thr:
        return 1 - distance / thr
    else:
        return 0
    return distance
# ========= 推理辅助 =========
def extract_score(model,template_img,target_img):
    
    with torch.no_grad():
        template_out,target_out = model(template_img, target_img)
        # template_out = F.normalize(template_out, p=2, dim=1)
        # target_out = F.normalize(target_out, p=2, dim=1)
        distances = F.pairwise_distance(template_out, target_out)  # [B]          #输出embedding
    return distances.item()

def match_with_cached_templates(model, target_canny_img, template_dir, topk=3, thr=0.75,score_limit=0.5):
    target_img = edge_to_binary(canny_img = target_canny_img)
    target_img_copy = target_img.copy()
    target_img = transform(target_img).unsqueeze(0).to(device)


    scores = []
    meta = []
    image_pare = []
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
            continue

        fpath = os.path.join(template_dir, fname)
        template_img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        ret, _ = cv2.threshold(template_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_threshold = max(0, int(ret * 0.5))  # 下限阈值设为Otsu阈值的50%
        high_threshold = min(255, int(ret * 1.5))  # 上限阈值设为Otsu阈值的150%
        # cv2.imshow("patch", patch)
        template_canny_img = cv2.Canny(template_img, low_threshold, high_threshold)
        template_canny_img = cv2.copyMakeBorder(template_canny_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        #腐蚀膨胀
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        template_canny_img = cv2.dilate(template_canny_img, kernel_dilate)
        template_canny_img = cv2.erode(template_canny_img, kernel_erode)
        # img_np = np.array(img)

        template_img = edge_to_binary(canny_img = template_canny_img)
        template_img_copy = template_img.copy()
        template_img = transform(template_img).unsqueeze(0).to(device)
        distence = extract_score(model, template_img, target_img)
        score = score_match(distence, thr)
        scores.append(score)
        cls_name = os.path.basename(os.path.dirname(fpath))  # 可按文件夹命名区分类别
        meta.append((cls_name, fname))  # 或 
        #meta.append((fname,)) #如果只需要文件名
        image_pare.append((template_img_copy,target_img_copy))

    scores = np.array(scores)
    # dists = 1 - scores
    dists = 1-scores
    idx = np.argsort(dists)[:topk]
    results = [(scores[i], *meta[i], image_pare[i]) if scores[i] >= score_limit else (0.0, "no match", "", "") for i in idx]
    # results = [(scores[i], *meta[i], image_pare[i]) for i in idx]
    return results

# ========= 主流程 =========
if __name__ == "__main__":
    # model = SiameseNet().to(device)
    # model.load_state_dict(torch.load(r"model_pth\jus_cnn_se\best_model_best_acc_0.9444_best_f1_0.9443_best_thresh_0.7970.pth", map_location=device))
    # model.eval()
    TripletLoss_model = TripletLoss().to(device)
    TripletLoss_model.load_state_dict(torch.load(r"model_pth\jus_cnn_se_Triplet_Loss_hard_no_normalize_L2\best_model_best_acc_328.1755_best_f1_0.9718_best_thresh_146.pth", map_location=device))
    branch = TripletLoss_model.mobilenetv3_like
    model = TripletNet_2_branch(branch).to(device)
    model.eval()


    # cv2.namedWindow("contour_mask", cv2.WINDOW_NORMAL)
    img_gray = cv2.imread(r"mubiao_2.bmp", cv2.IMREAD_GRAYSCALE)
    start_time = time.time()
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 5)
    img_bin = cv2.dilate(img_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_bin.shape
    result_img = img_gray.copy()
    if len(img_gray.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    for cnt in contours:
        if len(cnt) < 10: continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww / w > 0.9 and hh / h > 0.9: continue
        if cv2.contourArea(cnt) < 500: continue

        # mask = np.zeros_like(img_bin)
        # cv2.fillPoly(mask, [cnt], 255)
        # patch = mask[y:y + hh, x:x + ww]
        cut_img = img_gray[y:y + hh, x:x + ww]
        # cv2.imshow("cut_img", cut_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #区域直方图均衡化
        # cut_img = cv2.equalizeHist(cut_img)
        # cv2.imshow("cut_img", cut_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #otsu二值化
        ret, binary = cv2.threshold(cut_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("binary", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        low_threshold = max(0, int(ret * 0.5))  # 下限阈值设为Otsu阈值的50%
        high_threshold = min(255, int(ret * 1.5))  # 上限阈值设为Otsu阈值的150%
        # cv2.imshow("patch", patch)
        canny_img = cv2.Canny(cut_img, low_threshold, high_threshold)
        canny_img = cv2.copyMakeBorder(canny_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        # cv2.imshow("canny_img", canny_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #腐蚀膨胀
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canny_img = cv2.dilate(canny_img, kernel_dilate)
        canny_img = cv2.erode(canny_img, kernel_erode)
        # cv2.imshow("contour_mask", patch)
        # cv2.waitKey(0)
        # canny_img = Image.fromarray(canny_img.astype('uint8'))
        res = match_with_cached_templates(
            model, canny_img,
            template_dir=r"model_embeddings\model_img",
            topk=5, thr=900.1755, score_limit=0.3
        )
        print("Top Matches:")
        for dist, cls, name, image_pare in res:
            print(f"Class: {cls}, Image: {name}, Dist: {dist:.4f}")
        try:
            best_dict, cls_name, fname, image_pare = res[0]
            # cv2.imshow("cut_img", image_pare[0])
            # cv2.imshow("target_img", image_pare[1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        except:
            continue
        if best_dict > 0:
            image_pare1 = cv2.cvtColor(image_pare[0], cv2.COLOR_RGB2GRAY)
            image_pare2 = cv2.cvtColor(image_pare[1], cv2.COLOR_RGB2GRAY)
            _,_,theta_opt,scale_opt = shape_context_matching(image_pare1, image_pare2, n_points=80)
            theta = radian_to_degree(theta_opt)
            cv2.rectangle(result_img, (x, y), (x + ww, y + hh), (0, 255, 0))
            label = f"{fname}:{best_dict:.2f} angle:{theta:.2f} scale:{scale_opt:.2f}"
            # label = f"{fname}:{best_dict:.2f}"
            cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.2f}s") 
    cv2.imshow("result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
