import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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


def shape_context_matching_with_weighted_hausdorff(img1, img2, n_points=100, corner_radius=10):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    def sample_contour_points(img, n):
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

    def detect_corners_from_image(img_gray, max_corners=10, quality_level=0.1, min_distance=30):
        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=max_corners,
                                          qualityLevel=quality_level,
                                          minDistance=min_distance)
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2)

    def assign_weights(points, corners, radius):
        weights = np.ones(len(points), dtype=np.float32)
        if len(corners) == 0:
            return weights
        for i, pt in enumerate(points):
            dists = np.linalg.norm(corners - pt, axis=1)
            if np.any(dists <= radius):
                weights[i] = 1.0  # 角点邻域权重
        return weights

    def transform_points(points, theta, scale):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return scale * points @ R.T

    def weighted_hausdorff_distance(params, src, dst, weights_src, weights_dst):
        theta, scale = params
        transformed = transform_points(src, theta, scale)
        D = cdist(transformed, dst)
        d1 = np.min(D, axis=1)
        d1_weighted = d1 * weights_src
        d2 = np.min(D, axis=0)
        d2_weighted = d2 * weights_dst
        return max(np.max(d1_weighted), np.max(d2_weighted))
    # def weighted_hausdorff_distance(params, src, dst, weights_src, weights_dst):
    #     theta, scale = params
    #     transformed = transform_points(src, theta, scale)
        
    #     # 计算加权距离矩阵（权重作用于行和列）
    #     D = cdist(transformed, dst)
    #     D_weighted = D * weights_src[:, None] * weights_dst[None, :]  # 元素级加权
        
    #     # 计算双向最小加权距离
    #     d1 = np.min(D_weighted, axis=1)  # 每个src点到dst的最小加权距离
    #     d2 = np.min(D_weighted, axis=0)  # 每个dst点到src的最小加权距离
        
    #     return max(np.max(d1), np.max(d2))  # 对称Hausdorff距离
    # def weighted_hausdorff_distance(params, src, dst, weights_src, weights_dst):
    #     theta, scale = params
    #     transformed = transform_points(src, theta, scale)
    #     D = cdist(transformed, dst)

    #     # 每个源点到目标点集的最小距离，加权
    #     d1 = np.min(D, axis=1)
    #     d1_avg = np.average(d1, weights=weights_src)

    #     # 每个目标点到源点集的最小距离，加权
    #     d2 = np.min(D, axis=0)
    #     d2_avg = np.average(d2, weights=weights_dst)

    #     return (d1_avg + d2_avg) / 2  # 双向加权平均

    # 确保传入灰度图
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # 1. 采样轮廓点
    pts1 = sample_contour_points(img1, n_points).astype(np.float32)
    pts2 = sample_contour_points(img2, n_points).astype(np.float32)

    # 2. 计算实心区域质心中心化
    center1 = compute_centroid_mask_from_points(img1.shape, pts1)
    center2 = compute_centroid_mask_from_points(img2.shape, pts2)
    pts1_centered = pts1 - center1
    pts2_centered = pts2 - center2

    # 3. 从原图灰度图检测角点
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)
    corners1 = detect_corners_from_image(gray1)
    visualize_corners(gray1, corners1)
    corners2 = detect_corners_from_image(gray2)
    visualize_corners(gray2, corners2)
    pts1_restored = pts1_centered + center1
    pts2_restored = pts2_centered + center2

    # 4. 赋予轮廓点权重，角点邻域权重5，其他1
    weights1 = assign_weights(pts1_centered, corners1 - center1, corner_radius)
    weights1_restored = assign_weights(pts1_restored, corners1, corner_radius)
    visualize_weights_color(gray1, pts1_restored, weights1_restored)
    weights2 = assign_weights(pts2_centered, corners2 - center2, corner_radius)
    weights2_restored = assign_weights(pts2_restored, corners2, corner_radius)
    visualize_weights_color(gray2, pts2_restored, weights2_restored)
    # weights1 = weights1 / np.mean(weights1)
    # weights2 = weights2 / np.mean(weights2)
    # 5. 优化
    initial_params = [0.0, 1.0]
    bounds = [(-np.pi, np.pi), (0.8, 1.2)]
    result = minimize(weighted_hausdorff_distance, initial_params,
                      args=(pts1_centered, pts2_centered, weights1, weights2),
                      bounds=bounds, strategy='best1bin', maxiter=1000, tol=1e-6)
    theta_opt, scale_opt = result.x

    # 6. 应用变换恢复平移
    pts1_transformed = transform_points(pts1_centered, theta_opt, scale_opt) + center2

    # 7. 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Points")
    plt.scatter(pts1[:, 0], pts1[:, 1], c='blue', label='Template')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='red', label='Target')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Aligned Template (Weighted Hausdorff)")
    plt.scatter(pts1_transformed[:, 0], pts1_transformed[:, 1], c='green', label='Transformed Template')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='red', label='Target')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return pts1_transformed, pts2, theta_opt, scale_opt





if __name__ == '__main__':
    img1 = cv2.imread('org_muban\muban_1.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), -50, 0.9)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    # img2 = cv2.imread('target.png', 0)
    # _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    # _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 方法一：Shape Context 匹配点对并可视化
    _,_,theta_opt,scale_opt = shape_context_matching_with_weighted_hausdorff(img1, img2)
    print(theta_opt,scale_opt)