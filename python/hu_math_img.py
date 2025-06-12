import numpy as np
import cv2
import time
import os

def load_templates(template_dir):
    """载入模板图像并计算Hu矩特征"""
    template_features = []
    template_metadata = []
    
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.bmp')):
            # 读取模板图像
            img_path = os.path.join(template_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 二值化处理
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 计算Hu矩
            hu = compute_hu_moments(binary)
            
            # 存储特征和元数据
            template_features.append(hu)
            template_metadata.append((filename.split('.')[0], filename))
    
    return np.array(template_features), template_metadata

def compute_hu_moments(img):
    """计算图像的Hu矩特征"""
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments)
    # 对数变换增强稳定性
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu.flatten()

def match_with_hu(query_hu, template_features, template_metadata, threshold=30, topk=3):
    """使用Hu矩进行匹配"""
    # 计算距离（欧式距离）
    dists = np.linalg.norm(template_features - query_hu, axis=1)
    
    # 获取topk匹配结果
    topk_idx = np.argsort(dists)[:topk]
    results = [(dists[i], template_metadata[i][0], template_metadata[i][1]) if dists[i] < threshold 
               else (0, "无匹配", "") for i in topk_idx]
    return results

def process_image(image_path, template_features, template_metadata):
    """处理输入图像"""
    # 读取图像
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 5)
    
    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_binary = cv2.dilate(img_binary, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img_binary.shape[:2]
    
    # 准备结果图像
    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # 过滤小轮廓
        if len(contour) < 10:
            continue
            
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤接近图像边界的轮廓
        if w/img_w > 0.90 or h/img_h > 0.90:
            continue
            
        # 过滤小面积轮廓
        if cv2.contourArea(contour) < 500:
            continue
        
        # 创建轮廓掩模
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        contour_roi = mask[y:y+h, x:x+w]
        
        # 计算Hu矩特征
        query_hu = compute_hu_moments(contour_roi)
        
        # 进行匹配
        matches = match_with_hu(query_hu, template_features, template_metadata)
        
        # 绘制结果
        if matches and matches[0][0] > 0:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{matches[0][1]}: {matches[0][0]:.2f}"
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return result_img

if __name__ == "__main__":
    # 模板目录路径
    TEMPLATE_DIR = r"model_embeddings\model_img"
    # 测试图像路径
    TEST_IMAGE = "mubiao_1.bmp"
    
    # 1. 载入模板
    print("正在载入模板...")
    start_time = time.time()
    template_features, template_metadata = load_templates(TEMPLATE_DIR)
    print(f"载入完成，共 {len(template_metadata)} 个模板，耗时 {time.time()-start_time:.2f}秒")
    
    # 2. 处理测试图像
    print("\n处理测试图像...")
    start_time = time.time()
    result_img = process_image(TEST_IMAGE, template_features, template_metadata)
    print(f"处理完成，耗时 {time.time()-start_time:.2f}秒")
    
    cv2.namedWindow("匹配结果", cv2.WINDOW_NORMAL)
    # 3. 显示结果
    cv2.imshow("匹配结果", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()