import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import ransac
from skimage.transform import AffineTransform,SimilarityTransform
import matplotlib.pyplot as plt

def chi2_cost(histA, histB):
    eps = 1e-10  # 防止除0
    return 0.5 * np.sum((histA - histB) ** 2 / (histA + histB + eps))

def match_descriptors_hungarian_chi2(desc1, desc2):
    n1, n2 = len(desc1), len(desc2)
    dist_matrix = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            dist_matrix[i, j] = chi2_cost(desc1[i], desc2[j])

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matches = [(i, j, dist_matrix[i, j]) for i, j in zip(row_ind, col_ind)]
    return matches
# def chi2_cost(histA, histB):
#     eps = 1e-10  # 防止除0
#     return 0.5 * np.sum((histA - histB) ** 2 / (histA + histB + eps))

def match_descriptors_mutual_best_chi2(desc1, desc2):
    n1, n2 = len(desc1), len(desc2)
    cost_matrix = np.zeros((n1, n2), dtype=np.float32)

    # 构建卡方距离矩阵
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = chi2_cost(desc1[i], desc2[j])

    # 每个点在对方中的最小索引
    best_in_2_for_1 = np.argmin(cost_matrix, axis=1)  # A 中每个点匹配到 B 的哪个点
    best_in_1_for_2 = np.argmin(cost_matrix, axis=0)  # B 中每个点匹配到 A 的哪个点

    matches = []
    for i in range(n1):
        j = best_in_2_for_1[i]
        if best_in_1_for_2[j] == i:
            matches.append((i, j, cost_matrix[i, j]))

    return matches



def extract_contour_points(img, min_area=10):
    edges = cv2.Canny(img, 50, 150)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilate = cv2.dilate(edges, kernel_dilate)
    img_erode = cv2.erode(img_dilate, kernel_erode)
    contours, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            for pt in cnt:
                all_points.append(pt[0])
    return np.array(all_points)

def shape_context_descriptor(points, r_bins=5, theta_bins=1, r_inner=0.125, r_outer=1):
    if len(points) < 2:
        return None
    descriptors = []
    # 计算点之间距离矩阵
    dists = cdist(points, points)
    mean_dist = np.mean(dists)
    # 归一化距离矩阵
    r_array = dists / (mean_dist + 1e-8)
    # 角度矩阵
    angles = np.arctan2(points[:, None, 1] - points[None, :, 1], points[:, None, 0] - points[None, :, 0])  # shape NxN

    for i in range(len(points)):
        # 排除自身点距离和角度
        r = r_array[i]
        theta = (angles[i] + 2 * np.pi) % (2 * np.pi)

        # log空间分割半径
        r_log = np.log(r + 1e-8)
        r_bin_edges = np.linspace(np.log(r_inner), np.log(r_outer), r_bins + 1)
        r_bin_idx = np.digitize(r_log, r_bin_edges) - 1
        r_bin_idx = np.clip(r_bin_idx, 0, r_bins - 1)

        # 角度分割
        theta_bin_idx = np.floor(theta / (2 * np.pi) * theta_bins).astype(int)
        theta_bin_idx = np.clip(theta_bin_idx, 0, theta_bins - 1)

        # 生成直方图
        hist = np.zeros((r_bins, theta_bins))
        for rb, tb in zip(r_bin_idx, theta_bin_idx):
            hist[rb, tb] += 1

        hist[i // len(points) if i // len(points) < r_bins else r_bins-1, 0] = 0  # 排除自身点
        hist = hist.flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        descriptors.append(hist)

    return np.array(descriptors)
def match_descriptors_mutual_nearest(desc1, desc2):
    dist_matrix = cdist(desc1, desc2, metric='euclidean')

    # 每个desc1点找到desc2中距离最小的点
    nn_1to2 = np.argmin(dist_matrix, axis=1)
    # 每个desc2点找到desc1中距离最小的点
    nn_2to1 = np.argmin(dist_matrix, axis=0)

    matches = []
    for i, j in enumerate(nn_1to2):
        # 互选条件
        if nn_2to1[j] == i:
            dist = dist_matrix[i, j]
            matches.append((i, j, dist))

    return matches
def match_descriptors_hungarian(desc1, desc2):
    dist_matrix = cdist(desc1, desc2, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matches = []
    for i, j in zip(row_ind, col_ind):
        dist = dist_matrix[i, j]
        matches.append((i, j, dist))
    return matches

def geometric_verification(matches, keypoints1, keypoints2, residual_threshold=5):
    """
    matches: [(i, j, dist), ...] 双向匹配索引和距离
    keypoints1: Nx2，desc1对应的点坐标
    keypoints2: Mx2，desc2对应的点坐标
    residual_threshold: RANSAC的内点阈值（像素距离）

    返回：通过几何验证的匹配对列表（同matches格式）
    """
    if len(matches) < 3:
        return matches  # 点太少，直接返回

    src = np.array([keypoints1[i] for i, j, _ in matches])
    dst = np.array([keypoints2[j] for i, j, _ in matches])

    model_robust, inliers = ransac(
        (src, dst),
        SimilarityTransform,
        min_samples=3,
        residual_threshold=residual_threshold,
        max_trials=1000
    )

    inlier_matches = [matches[k] for k in range(len(matches)) if inliers[k]]
    return inlier_matches
def ransac_affine_verification(matches, pts1, pts2, threshold=1):
    if len(matches) < 3:
        return None, None
    src = np.array([pts1[m[0]] for m in matches])
    dst = np.array([pts2[m[1]] for m in matches])

    model, inliers = ransac((src, dst),
                            SimilarityTransform,
                            min_samples=2,
                            residual_threshold=threshold,
                            max_trials=10000)
    inliers = np.array(inliers, dtype=bool)
    inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    return model, inlier_matches

def rotation_angle_from_affine(matrix):
    # affine matrix:
    # [a, b, tx]
    # [c, d, ty]
    # rotation = atan2(c, a)
    angle_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_matches(img1, img2, pts1, pts2, matches, inliers=None):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for idx, (i, j, dist) in enumerate(matches):
        pt1 = tuple(pts1[i].astype(int))
        pt2 = tuple((pts2[j] + np.array([w1, 0])).astype(int))
        color = (0, 255, 0) if (inliers is not None and inliers[idx]) else (0, 0, 255)
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    cv2.imshow('Matches', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1 = cv2.imread('model_embeddings\model_img\moban_1.jpg', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 15, 1)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

    pts1 = extract_contour_points(img1)
    pts2 = extract_contour_points(img2)

    desc1 = shape_context_descriptor(pts1)
    desc2 = shape_context_descriptor(pts2)

    # matches = match_descriptors_mutual_nearest(desc1, desc2)
    # matches = match_descriptors_hungarian(desc1, desc2)
    matches = match_descriptors_hungarian_chi2(desc1, desc2)
    #matches = match_descriptors_mutual_best_chi2(desc1, desc2)
    print(f"Total matches (Hungarian): {len(matches)}")

    model, inlier_matches = ransac_affine_verification(matches, pts1, pts2)
    if model is None:
        print("RANSAC failed, not enough matches")
        return

    print(f"Inlier matches after RANSAC: {len(inlier_matches)}")
    angle = rotation_angle_from_affine(model.params)
    print(f"Estimated rotation angle (degrees): {angle:.2f}")

    inliers_mask = [m in inlier_matches for m in matches]
    draw_matches(img1, img2, pts1, pts2, matches, inliers=inliers_mask)

if __name__ == '__main__':
    main()
