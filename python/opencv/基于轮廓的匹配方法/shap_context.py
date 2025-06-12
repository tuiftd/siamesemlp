# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes

# -----------------------------
# 方法一：Shape Context + 匈牙利 + Procrustes
# -----------------------------
def get_rotation_angle(mtx1, mtx2):
    # 假设是2D情况，取旋转矩阵的左上2x2部分
    R = mtx2[:2, :2]
    # 计算旋转角度（弧度）
    angle = np.arctan2(R[1, 0], R[0, 0])
    # 转换为角度
    angle_deg = np.degrees(angle)
    return angle_deg

# 在你的函数最后添加：
def shape_context_matching(img1, img2, n_points=100):
    def sample_contour_points(img, n):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        idx = np.round(np.linspace(0, len(cnt) - 1, n)).astype(int)
        return cnt[idx, 0, :]

    def compute_histogram(points, bins_r=5, bins_theta=36):
        n = len(points)
        histograms = []
        for i in range(n):
            r = points - points[i]
            theta = np.arctan2(r[:, 1], r[:, 0]) % (2 * np.pi)
            d = np.linalg.norm(r, axis=1)
            d[d == 0] = 1e-5
            r_bin = np.floor(np.log(d / d.min()) / np.log(d.max() / d.min() + 1e-5) * bins_r).astype(int)
            t_bin = np.floor(theta / (2 * np.pi) * bins_theta).astype(int)
            r_bin = np.clip(r_bin, 0, bins_r - 1)
            t_bin = np.clip(t_bin, 0, bins_theta - 1)
            hist = np.zeros((bins_r, bins_theta))
            for rb, tb in zip(r_bin, t_bin):
                hist[rb, tb] += 1
            histograms.append(hist.flatten())
        return np.array(histograms)

    def chi2_cost(h1, h2):
        cost = np.zeros((h1.shape[0], h2.shape[0]))
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                num = (h1[i] - h2[j])**2
                denom = h1[i] + h2[j] + 1e-5
                cost[i, j] = 0.5 * np.sum(num / denom)
        return cost

    pts1 = sample_contour_points(img1, n_points)
    pts2 = sample_contour_points(img2, n_points)
    h1 = compute_histogram(pts1)
    h2 = compute_histogram(pts2)
    cost = chi2_cost(h1, h2)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pts1 = pts1[row_ind]
    matched_pts2 = pts2[col_ind]

    mtx1, mtx2, disparity = procrustes(matched_pts1, matched_pts2)
    return matched_pts1, matched_pts2, mtx1, mtx2


# -----------------------------
# 方法二：基于距离变换的轮廓匹配
# -----------------------------
def distance_transform_match(template_bin, target_bin, angles=np.linspace(-15, 15, 31), scales=[1.0]):
    template_cnts, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    template_pts = template_cnts[0][:, 0, :]

    dist_map = cv2.distanceTransform(255 - target_bin, cv2.DIST_L2, 5)
    h, w = target_bin.shape

    best_score = float('inf')
    best_params = None

    for scale in scales:
        scaled = template_pts * scale
        for angle in angles:
            theta = np.deg2rad(angle)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated = scaled @ rot.T
            for dx in range(-w//4, w//4, 10):
                for dy in range(-h//4, h//4, 10):
                    trans = rotated + np.array([dx + w//2, dy + h//2])
                    trans = trans.astype(int)
                    valid = (0 <= trans[:, 0]) & (trans[:, 0] < w) & (0 <= trans[:, 1]) & (trans[:, 1] < h)
                    if np.sum(valid) == 0:
                        continue
                    values = dist_map[trans[valid][:, 1], trans[valid][:, 0]]
                    score = np.mean(values)
                    if score < best_score:
                        best_score = score
                        best_params = (scale, angle, dx, dy)
    return best_params


# -----------------------------
# 示例入口
# -----------------------------
if __name__ == '__main__':
    img1 = cv2.imread('org_muban\muban_3.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 80, 1)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    # img2 = cv2.imread('target.png', 0)
    _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 方法一：Shape Context 匹配点对并可视化
    pts1, pts2, mtx1, mtx2 = shape_context_matching(bin1, bin2)
    angle = get_rotation_angle(mtx1, mtx2)
    print("Best Match Params (Shape Context):", angle)
    vis = np.hstack([cv2.cvtColor(bin1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bin2, cv2.COLOR_GRAY2BGR)])
    offset = bin1.shape[1]
    for p1, p2 in zip(pts1, pts2):
        cv2.line(vis, tuple(p1), (p2[0] + offset, p2[1]), (0, 255, 0), 1)
    cv2.imshow("Shape Context Matching", vis)
    cv2.waitKey(0)

    # # 方法二：距离变换搜索最佳角度和位置
    # best_scale, best_angle, best_dx, best_dy = distance_transform_match(bin1, bin2)
    # print("Best Match Params (Distance Transform):", best_scale, best_angle, best_dx, best_dy)
