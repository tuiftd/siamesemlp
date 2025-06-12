import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import time
def radian_to_degree(radian):
    """
    将弧度转换为角度
    :param radian: 弧度值
    :return: 对应的角度值
    """
    degree = math.degrees(radian)
    return degree

def visualize_corners(img_gray, corners):
    # 将角点绘制在灰度图像上
    img_copy = img_gray.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_copy, (int(x), int(y)), radius=3, color=255, thickness=-1)
    cv2.imshow('Corners', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_weights_color(img_gray, points, weights, max_weight=100.0):
    # 创建一个与灰度图像大小相同的彩色图像，用于显示权重
    weight_img = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    
    # 将权重值映射到颜色空间
    for pt, weight in zip(points, weights):
        x, y = pt.astype(int)
        # 使用Blue到Red的颜色映射
        weight_img[y, x] = np.array([int(weight / max_weight * 255), 0, int((1 - weight / max_weight) * 255)])
    
    # 显示结果图像
    cv2.imshow('Weight Map (Color)', weight_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shape_context_matching(img1, img2, n_points=100, sigma=2):
    import cv2
    import numpy as np
    import time
    from scipy.spatial.distance import cdist
    from scipy.optimize import differential_evolution
    import math

    def sample_contour_points(img, n):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        idx = np.round(np.linspace(0, len(cnt) - 1, n)).astype(int)
        return cnt[idx, 0, :]

    def compute_centroid_mask_from_points(img_shape, points):
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(mask, [points.astype(np.int32)], -1, 255, thickness=-1)
        M = cv2.moments(mask)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return np.array([cx, cy], dtype=np.float32)

    def transform_points(points, theta, scale):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return scale * points @ R.T

    def generate_gaussian_weightmap(points, img_shape, sigma=5):
        heatmap = np.zeros(img_shape, dtype=np.float32)
        x_coords, y_coords = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
        for (px, py) in points:
            if 0 <= px < img_shape[1] and 0 <= py < img_shape[0]:
                dx = x_coords - px
                dy = y_coords - py
                dist_sq = dx**2 + dy**2
                kernel = np.exp(-dist_sq / (2 * sigma**2))
                heatmap += kernel
        heatmap /= np.max(heatmap)
        return heatmap

    def query_weights_from_heatmap(points, heatmap):
        h, w = heatmap.shape
        weights = []
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                weights.append(heatmap[y, x])
            else:
                weights.append(0)
        return np.array(weights, dtype=np.float32)

    def weighted_hausdorff_distance(params, src, dst, weights_src, weights_dst):
        theta, scale = params
        transformed = transform_points(src, theta, scale)
        D = cdist(transformed, dst)
        d1 = np.min(D, axis=1)
        d1_weighted = d1 * weights_src
        d2 = np.min(D, axis=0)
        d2_weighted = d2 * weights_dst
        return max(np.max(d1_weighted), np.max(d2_weighted))

    # 轮廓点提取与居中
    pts1 = sample_contour_points(img1, n_points).astype(np.float32)
    pts2 = sample_contour_points(img2, n_points).astype(np.float32)

    center1 = compute_centroid_mask_from_points(img1.shape, pts1)
    center2 = compute_centroid_mask_from_points(img2.shape, pts2)

    pts1_centered = pts1 - center1
    pts2_centered = pts2 - center2

    # ⬇️ 使用高斯热力图方式生成权重 ⬇️
    heatmap1 = generate_gaussian_weightmap(pts1_centered + center1, img1.shape, sigma=sigma)
    heatmap2 = generate_gaussian_weightmap(pts2_centered + center2, img2.shape, sigma=sigma)

    weights1 = query_weights_from_heatmap(pts1_centered + center1, heatmap1)
    weights2 = query_weights_from_heatmap(pts2_centered + center2, heatmap2)

    # ⬇️ Hausdorff优化匹配 ⬇️
    bounds = [(-np.pi, np.pi), (0.3, 1.9)]
    start_time = time.time()
    result = differential_evolution(
        weighted_hausdorff_distance,
        bounds,
        args=(pts1_centered, pts2_centered, weights1, weights2),
        strategy='best1bin',
        maxiter=40,
        tol=1e-3,
        polish=True
    )
    theta_opt, scale_opt = result.x
    end_time = time.time()
    print("Optimization time: {:.2f}s".format(end_time - start_time))

    # 应用变换 + 平移还原
    pts1_transformed = transform_points(pts1_centered, theta_opt, scale_opt) + center2

    return pts1_transformed, pts2, theta_opt, scale_opt

if __name__ == '__main__':
    img1 = cv2.imread('org_muban\muban_5.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 59,0.7)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    img1 = cv2.imread('model_embeddings\model_img\moban_5.jpg', 0)
    # img2 = cv2.imread('target.png', 0)
    _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 方法一：Shape Context 匹配点对并可视化
    _,_,theta_opt,scale_opt = shape_context_matching(bin1, bin2, n_points=70)
    theta = radian_to_degree(theta_opt)
    print(theta,scale_opt)