import numpy as np
import cv2
import time
import os

def load_template_contours(template_dir):
    """载入模板轮廓数据"""
    template_contours = []
    template_metadata = []
    
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.bmp')):
            # 读取模板图像
            img_path = os.path.join(template_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 二值化处理
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             # 创建空白图像
            # img_np = np.zeros_like(img_np)

            # # 填充所有轮廓（内部和外部）
            # cv2.drawContours(img_np, contours, -1, 255, cv2.FILLED)
            if contours:
                # 只取面积最大的轮廓
                main_contour = max(contours, key=cv2.contourArea)
                template_contours.append(main_contour)
                template_metadata.append((filename.split('.')[0], filename))
    
    return template_contours, template_metadata

def match_with_shape(query_contour, template_contours, template_metadata, threshold=0.5, topk=3):
    """使用cv2.matchShapes进行形状匹配"""
    distances = []
    for t_contour in template_contours:
        # 计算形状距离（值越小表示越相似）
        dist = cv2.matchShapes(query_contour, t_contour, cv2.CONTOURS_MATCH_I3, 0)
        distances.append(dist)
    
    distances = np.array(distances)
    # 获取topk匹配结果（按距离升序排序）
    topk_idx = np.argsort(distances)[:topk]
    results = [(distances[i], template_metadata[i][0], template_metadata[i][1]) if distances[i] < threshold 
               else (0, "无匹配", "") for i in topk_idx]
    return results

def process_image(image_path, template_contours, template_metadata):
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
        
        # 进行形状匹配
        matches = match_with_shape(contour, template_contours, template_metadata)
        
        # 绘制结果
        if matches and matches[0][0] > 0:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{matches[0][1]}: {matches[0][0]:.2f}"
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return result_img

if __name__ == "__main__":
    # 模板目录路径
    TEMPLATE_DIR =r"model_embeddings\model_img"
    # 测试图像路径
    TEST_IMAGE = "mubiao_1.bmp"
    
    # 1. 载入模板轮廓
    print("正在载入模板轮廓...")
    start_time = time.time()
    template_contours, template_metadata = load_template_contours(TEMPLATE_DIR)
    print(f"载入完成，共 {len(template_metadata)} 个模板，耗时 {time.time()-start_time:.2f}秒")
    
    # 2. 处理测试图像
    print("\n处理测试图像...")
    start_time = time.time()
    result_img = process_image(TEST_IMAGE, template_contours, template_metadata)
    print(f"处理完成，耗时 {time.time()-start_time:.2f}秒")
    cv2.namedWindow("形状匹配结果", cv2.WINDOW_NORMAL)
    # 3. 显示结果
    cv2.imshow("形状匹配结果", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()