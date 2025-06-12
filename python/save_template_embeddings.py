# save_template_embeddings.py
import os
import torch
import numpy as np
from model import HuNetAttn_1D as TripletNet
from tqdm import tqdm
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_log_hu_moments(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 找到轮廓
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白图像
    img_np = np.zeros_like(img_np)

    # 填充所有轮廓（内部和外部）
    cv2.drawContours(img_np, contours, -1, 255, cv2.FILLED)
    # for cnt in contours:
    # # 多边形拟合（调整epsilon值控制拟合精度）
    #     epsilon = 0.005 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        
    #     # 绘制填充多边形（颜色可自定义）
    #     cv2.fillPoly(img_np, [approx], color=255)  # 填充白色
        
    cv2.imshow('img', img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    m = cv2.moments(img_np)
    hu = cv2.HuMoments(m)
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu.flatten()

def extract_embedding(model, image_path):
    hu_feature = compute_log_hu_moments(image_path)
    input_tensor = torch.tensor(hu_feature, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(input_tensor)
    return embedding.cpu().numpy().flatten()

def save_all_template_embeddings(model, template_dir, output_root=r"model_embeddings\embeddings"):
    embeddings = []
    metadata = []

    for cls_name in tqdm(os.listdir(template_dir), desc="Processing Classes"):
        class_path = os.path.join(template_dir, cls_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg','.bmp', '.gif')):
                img_path = os.path.join(class_path, img_name)
                try:
                    emb = extract_embedding(model, img_path)
                    embeddings.append(emb)
                    metadata.append((cls_name, img_name))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    embeddings = np.array(embeddings)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    embeddings_path = os.path.join(output_root, "template_embeddings.npy")
    metadata_path = os.path.join(output_root, "template_metadata.npy")
    np.save(embeddings_path, embeddings)
    np.save(metadata_path, metadata)
    print(f"Saved {len(embeddings)} embeddings to {output_root}.")
    print(f"Saved {len(embeddings)} embeddings.")

if __name__ == "__main__":
    model = TripletNet().to(device)
    model.load_state_dict(torch.load(r"model_pth\tripletmlp\triplet_action_hu_trained_1.pth"))
    model.eval()

    template_root = r"model_embeddings"  # 你的模板路径
    save_all_template_embeddings(model, template_root)
