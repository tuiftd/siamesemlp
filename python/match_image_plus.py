# match_with_cached_templates.py
import numpy as np
from model import TripletNet_attention as TripletNet
import torch
import cv2
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_log_hu_moments(image_path=None, img=None):
    if img is None:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
    else:
        img_np = img
    #  
    # for cnt in contours:
    # # 多边形拟合（调整epsilon值控制拟合精度）
    #     epsilon = 0.005 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        
    #     # 绘制填充多边形（颜色可自定义）
    #     cv2.fillPoly(img_np, [approx], color=255)  # 填充白色
    # cv2.imshow("binary", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    m = cv2.moments(img_np)
    hu = cv2.HuMoments(m)
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu.flatten()

def extract_embedding(model, image_path=None, img=None):
    if img is None:
        hu_feature = compute_log_hu_moments(image_path)
    else:
        hu_feature = compute_log_hu_moments(img=img)
    input_tensor = torch.tensor(hu_feature, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(input_tensor)
    return embedding.cpu().numpy().flatten()

def match_with_cached_templates(model, query_image_path=None, query_img=None, emb_file=r"model_embeddings\embeddings\template_embeddings.npy", meta_file=r"model_embeddings\embeddings\template_metadata.npy", topk=3):
    # 加载缓存的模板特征和元数据
    template_embeddings = np.load(emb_file)
    template_metadata = np.load(meta_file, allow_pickle=True)
    
    # 提取查询图像特征
    start_time = time.time()
    if query_img is None:
        query_embedding = extract_embedding(model, query_image_path)
    else:
        query_embedding = extract_embedding(model, img=query_img)
    end_time = time.time()
    print("Time used for extracting query embedding:", end_time - start_time)
    # 计算欧氏距离
    dists = np.linalg.norm(template_embeddings - query_embedding, axis=1)
    topk_idx = np.argsort(dists)[:topk]
    results = [(dists[i], template_metadata[i][0], template_metadata[i][1]) if dists[i] < 30 else (0, "no match", "") for i in topk_idx]
    return results

if __name__ == "__main__":
    model = TripletNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(r"model_pth\tripletmlp\triplet_action_hu_trained_1.pth"))
    model.eval()
    cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    start_time = time.time()
    chack_img_path = r"mubiao_1.bmp"
    img_gray = cv2.imread(chack_img_path, cv2.IMREAD_GRAYSCALE)
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_binary = cv2.dilate(img_binary, kernel)

    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img_binary.shape[:2]
    print(img_w, img_h)
    count = 0
    
    # Create a copy of the original image to draw on
    result_img = img_gray.copy()
    if len(img_gray.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # 获取轮廓的面积
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if len(contour) < 10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w/img_w > 0.90 and h/img_h > 0.90:  # 防止图片边框被检索到
            continue
        area = cv2.contourArea(contour)
        # 面积小于1000的轮廓忽略
        if area < 500:
            continue
        print(x, y, w, h)
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts=[contour], color=255)
        # 裁剪保留轮廓x,y,w,h
        contour_mask = mask[y:y+h, x:x+w]
        # cv2.imshow("contour_mask", contour_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        results = match_with_cached_templates(model=model, query_img=contour_mask, topk=5)
        print("Top Matches:")
        for dist, cls, name in results:
            print(f"Class: {cls}, Image: {name}, Distance: {dist:.4f}")
        
        # Draw the top result if distance is not zero
        if results and results[0][0] > 0:
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Put text label (class name and distance)
            label = f"{results[0][1]}: {results[0][0]:.2f}"
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Show the final result image with all matches
    cv2.namedWindow("Matching Results", cv2.WINDOW_NORMAL)
    cv2.imshow("Matching Results", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print("Time used:", end_time - start_time)