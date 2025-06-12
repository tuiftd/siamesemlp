import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt


def extract_contour_points(img, min_area=10):
    """提取轮廓点"""
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    all_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            for pt in cnt:
                all_points.append(pt[0])
    return np.array(all_points)


def polar_histogram(points, center=None, radius_bins=6, angle_bins=12):
    """生成轮廓点的极坐标直方图描述子"""
    if len(points) < 5:
        return None

    if center is None:
        center = np.mean(points, axis=0)

    relative = points - center
    r = np.linalg.norm(relative, axis=1)
    theta = (np.arctan2(relative[:, 1], relative[:, 0]) + 2 * np.pi) % (2 * np.pi)

    r_norm = r / (r.max() + 1e-8)  # normalize radius

    hist = np.zeros((radius_bins, angle_bins), dtype=np.float32)
    for ri, ai in zip(
        np.floor(r_norm * radius_bins).astype(int),
        np.floor(theta / (2 * np.pi) * angle_bins).astype(int),
    ):
        ri = min(ri, radius_bins - 1)
        ai = min(ai, angle_bins - 1)
        hist[ri, ai] += 1

    hist = hist.flatten()
    return hist / (np.linalg.norm(hist) + 1e-6)


def extract_descriptors(img, stride=8, window=24):
    """滑动窗口提取极坐标直方图描述子"""
    h, w = img.shape
    descriptors = []
    positions = []

    for y in range(0, h - window, stride):
        for x in range(0, w - window, stride):
            patch = img[y : y + window, x : x + window]
            contour_pts = extract_contour_points(patch)
            if len(contour_pts) == 0:
                continue
            contour_pts = contour_pts + np.array([x, y])  # 变为整图坐标
            center = np.mean(contour_pts, axis=0)
            desc = polar_histogram(contour_pts, center)
            if desc is not None:
                descriptors.append(desc)
                positions.append(center)
    return np.array(descriptors), np.array(positions)


def bidirectional_match(desc1, desc2, pos1, pos2, threshold=0.85):
    """基于余弦相似度的双向匹配"""
    sim_matrix = 1 - cdist(desc1, desc2, 'cosine')
    matches = []
    for i in range(len(desc1)):
        j = np.argmax(sim_matrix[i])
        i_back = np.argmax(sim_matrix[:, j])
        if i_back == i and sim_matrix[i, j] > threshold:
            matches.append((pos1[i], pos2[j], sim_matrix[i, j]))
    return matches


def ransac_affine_verification(matches):
    """使用RANSAC筛选内点并估计仿射变换"""
    if len(matches) < 3:
        return matches, []

    src = np.array([m[0] for m in matches])
    dst = np.array([m[1] for m in matches])

    model, inliers = ransac(
        (src, dst),
        AffineTransform,
        min_samples=3,
        residual_threshold=5,
        max_trials=1000,
    )
    inliers = np.array(inliers, dtype=bool)
    inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    outliers = [matches[i] for i in range(len(matches)) if not inliers[i]]
    return inlier_matches, outliers


def draw_matches(img1, img2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    result[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for p1, p2, sim in matches:
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        cv2.line(result, p1, p2, (0, 255, 0), 1)
        cv2.circle(result, p1, 2, (255, 0, 0), -1)
        cv2.circle(result, p2, 2, (255, 0, 0), -1)

    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # 加载图像
    img1 = cv2.imread(r'org_muban\muban_1.bmp', 0)
    print(f"Image 1 shape: {img1.shape}")
    M = cv2.getRotationMatrix2D((img1.shape[1] // 2, img1.shape[0] // 2), 30, 0.8)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

    # 特征提取
    print("Extracting descriptors from target...")
    desc1, pos1 = extract_descriptors(img1)
    print("Extracting descriptors from template...")
    desc2, pos2 = extract_descriptors(img2)

    print(f"Target features: {len(desc1)}, Template features: {len(desc2)}")

    # 匹配
    raw_matches = bidirectional_match(desc1, desc2, pos1, pos2, threshold=0.85)

    # RANSAC 验证
    verified_matches, _ = ransac_affine_verification(raw_matches)

    print(f"Verified matches: {len(verified_matches)}")
    draw_matches(img1, img2, verified_matches)


if __name__ == "__main__":
    main()
