import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment


def shape_context_matching(img1, img2, n_points=200):
    def sample_contour_points(img, n):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        idx = np.round(np.linspace(0, len(cnt) - 1, n)).astype(int)
        return cnt[idx, 0, :]

    def compute_histogram(points, bins_r=1, bins_theta=1):
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

    # 对齐
    mtx1, mtx2, disparity = procrustes(matched_pts1, matched_pts2)

    # 可视化原始匹配点
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Matched Points")
    plt.scatter(pts1[:, 0], pts1[:, 1], c='blue', label='Template')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='red', label='Target')
    for p1, p2 in zip(matched_pts1, matched_pts2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.legend()

    # 可视化 Procrustes 对齐结果
    plt.subplot(1, 2, 2)
    plt.title("After Procrustes Alignment")
    plt.scatter(mtx1[:, 0], mtx1[:, 1], c='blue', label='Transformed Template')
    plt.scatter(mtx2[:, 0], mtx2[:, 1], c='red', label='Target')
    for p1, p2 in zip(mtx1, mtx2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return matched_pts1, matched_pts2, mtx1, mtx2

if __name__ == '__main__':
    img1 = cv2.imread('org_muban\muban_2.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 80, 1)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    # img2 = cv2.imread('target.png', 0)
    _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 方法一：Shape Context 匹配点对并可视化
    pts1, pts2, mtx1, mtx2 = shape_context_matching(bin1, bin2)
    vis = np.hstack([cv2.cvtColor(bin1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bin2, cv2.COLOR_GRAY2BGR)])
    offset = bin1.shape[1]
    for p1, p2 in zip(pts1, pts2):
        cv2.line(vis, tuple(p1), (p2[0] + offset, p2[1]), (0, 255, 0), 1)
    cv2.imshow("Shape Context Matching", vis)
    cv2.waitKey(0)