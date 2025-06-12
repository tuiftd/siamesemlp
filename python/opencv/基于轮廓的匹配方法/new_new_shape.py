import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def shape_context_matching(img1, img2, n_points=100):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.spatial.distance import directed_hausdorff

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

    def hausdorff_distance(A, B):
        return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

    def cost_function(params, src, dst):
        theta, scale = params
        transformed = transform_points(src, theta, scale)
        return hausdorff_distance(transformed, dst)

    # 提取轮廓点
    pts1 = sample_contour_points(img1, n_points).astype(np.float32)
    pts2 = sample_contour_points(img2, n_points).astype(np.float32)

    # 以绘制实心区域计算质心
    center1 = compute_centroid_mask_from_points(img1.shape, pts1)
    center2 = compute_centroid_mask_from_points(img2.shape, pts2)

    pts1_centered = pts1 - center1
    pts2_centered = pts2 - center2

    # 优化 scale 和角度 theta，保持质心对齐
    initial_params = [0.0, 1.0]
    bounds = [(-np.pi, np.pi), (0.8, 1.2)]
    result = minimize(cost_function, initial_params, args=(pts1_centered, pts2_centered), bounds=bounds)
    theta_opt, scale_opt = result.x

    # 应用最优变换并还原平移
    pts1_transformed = transform_points(pts1_centered, theta_opt, scale_opt) + center2

    # 可视化对齐前后结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Points")
    plt.scatter(pts1[:, 0], pts1[:, 1], c='blue', label='Template')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='red', label='Target')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Aligned Template")
    plt.scatter(pts1_transformed[:, 0], pts1_transformed[:, 1], c='green', label='Transformed Template')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='red', label='Target')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return pts1_transformed, pts2, theta_opt, scale_opt

if __name__ == '__main__':
    img1 = cv2.imread('org_muban\muban_2.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 80, 0.9)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    # img2 = cv2.imread('target.png', 0)
    _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 方法一：Shape Context 匹配点对并可视化
    _,_,theta_opt,scale_opt = shape_context_matching(bin1, bin2)
    print(theta_opt,scale_opt)