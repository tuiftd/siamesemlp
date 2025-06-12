import cv2
import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cdist
from skimage.transform import EuclideanTransform, estimate_transform
from skimage.measure import ransac

def bidirectional_bf_match(desc1, desc2, pos1, pos2, threshold=0.7):
    """双向暴力匹配"""
    # 计算余弦相似度矩阵
    sim_matrix = 1 - cdist(desc1, desc2, 'cosine')
    
    # 正向匹配
    matches_1to2 = np.argmax(sim_matrix, axis=1)
    sim_1to2 = np.max(sim_matrix, axis=1)
    
    # 反向匹配
    matches_2to1 = np.argmax(sim_matrix, axis=0)
    sim_2to1 = np.max(sim_matrix, axis=0)
    
    # 筛选双向一致匹配
    valid_matches = []
    for i in range(len(matches_1to2)):
        j = matches_1to2[i]
        if matches_2to1[j] == i and sim_1to2[i] > threshold and sim_2to1[j] > threshold:
            valid_matches.append((i, j, (sim_1to2[i] + sim_2to1[j]) / 2))
    
    # 转换为坐标对
    matched_pairs = []
    for i, j, sim in valid_matches:
        pt1 = pos1[i]
        pt2 = pos2[j]
        matched_pairs.append((pt1, pt2, sim))
    
    return matched_pairs

def ransac_geometric_verification(matched_pairs, distance_threshold=5, min_samples=3):
    """RANSAC几何验证"""
    if len(matched_pairs) < min_samples:
        return matched_pairs, np.array([])
    
    src_pts = np.array([m[0] for m in matched_pairs])
    dst_pts = np.array([m[1] for m in matched_pairs])
    
    # 估计欧式变换
    transform, inliers = ransac(
    (src_pts, dst_pts),
    EuclideanTransform,
    min_samples=3,
    residual_threshold=2,
    max_trials=1000
)
    
    # 筛选内点
    verified_matches = [matched_pairs[i] for i in range(len(matched_pairs)) if inliers[i]]
    outlier_matches = [matched_pairs[i] for i in range(len(matched_pairs)) if not inliers[i]]
    
    return verified_matches, outlier_matches

def visualize_matches(img1, img2, verified_matches, outlier_matches=None):
    """可视化匹配结果"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建拼接图像
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    result[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # 绘制内点匹配（绿色）
    for pt1, pt2, _ in verified_matches:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + w1), int(pt2[1])
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(result, (x1, y1), 2, (0, 255, 0), -1)
        cv2.circle(result, (x2, y2), 2, (0, 255, 0), -1)
    
    # 绘制外点匹配（红色，如果存在）
    if outlier_matches:
        for pt1, pt2, _ in outlier_matches:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0] + w1), int(pt2[1])
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # 显示图像
    cv2.imshow("Feature Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_fourier_descriptor(contour, m=10):
    """提取轮廓的傅里叶描述子（旋转/平移/缩放不变）"""
    if len(contour) < 5:
        return None  # 返回None表示无效特征
    
    centroid = np.mean(contour, axis=0)
    contour = contour - centroid
    z = contour[:, 0] + 1j * contour[:, 1]
    Z = np.abs(fft(z))
    if np.max(Z[1:]) > 1e-5:
        Z = Z / np.linalg.norm(Z[1:])
    return Z[:m]

def fulling_edge(img):
    """填充边缘"""
    # 膨胀腐蚀
    img_copy = img.copy()
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilate = cv2.dilate(img_copy, kernel_dilate)
    img_erode = cv2.erode(img_dilate, kernel_erode)
    contours,_ = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # 面积大于1000的轮廓才进行填充
                #跳过
                continue
            cv2.drawContours(img, [contour], 0, 255, -1)
    return img
    

def sliding_window_fourier_descriptors(edge_img, window_size=16, stride=4, m=10):
    """滑动窗口提取傅里叶描述子 + 可视化标记"""
    # edge_img = cv2.Canny(edge_img, 50, 150)  # 边缘检测
    h, w = edge_img.shape
    descriptors = []
    positions = []
    
    # 创建可视化图像（在原边缘图上标记）
    vis_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = edge_img[y:y+window_size, x:x+window_size]
            contours, _ = cv2.findContours(window, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = np.vstack([c.reshape(-1, 2) for c in contours])
                fd = extract_fourier_descriptor(contour, m)
                if fd is not None and fd.shape[0] == m:
                    descriptors.append(fd)
                    positions.append((x + window_size//2, y + window_size//2))
                    # 在区块中心画绿色圆点标记有效特征
                    cv2.circle(vis_img, (x + window_size//2, y + window_size//2), 
                              2, (0, 255, 0), -1)
            # 画区块边界（黄色矩形）
            # cv2.rectangle(vis_img, (x, y), (x+window_size, y+window_size), 
            #              (0, 255, 255), 1)
    
    # 显示特征提取可视化结果
    cv2.imshow("Feature Extraction Visualization", vis_img)
    cv2.waitKey(0)  # 短暂显示但不阻塞
    return np.array(descriptors), np.array(positions)

# 其余函数保持不变（bidirectional_bf_match, ransac_geometric_verification, visualize_matches）

def main():
    img1 = cv2.imread(r'org_muban\muban_2.bmp', cv2.IMREAD_GRAYSCALE)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 35, 0.8)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    
    
    edges1 = cv2.Canny(img1, 50, 150)
    edges2 = cv2.Canny(img2, 50, 150)
    edges1 = fulling_edge(edges1)
    edges2 = fulling_edge(edges2)
    
    # 提取特征并显示可视化
    print("Processing target image...")
    desc1, pos1 = sliding_window_fourier_descriptors(edges1)
    print("Processing template image...")
    desc2, pos2 = sliding_window_fourier_descriptors(edges2)
    
    matched_pairs = bidirectional_bf_match(desc1, desc2, pos1, pos2, threshold=0.65)
    verified_matches, _ = ransac_geometric_verification(matched_pairs)
    
    print(f"Final matched pairs: {len(verified_matches)}")
    visualize_matches(img1, img2, verified_matches)
    
    # 保存可视化结果
    cv2.imwrite("target_features.png", cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("template_features.png", cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR))

if __name__ == "__main__":
    main()