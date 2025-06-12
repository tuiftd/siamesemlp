import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
import time
def extract_contours(img, min_area=10):
    edges = cv2.Canny(img, 50, 150)
    cv2.imshow('edges', edges)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilate = cv2.dilate(edges, kernel_dilate)
    img_erode = cv2.erode(img_dilate, kernel_erode)
    contours, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered


def shape_context_descriptor(points, nbins_r=5, nbins_theta=12, r_inner=0.125, r_outer=2):
    if len(points) < 5:
        return None
    centroid = np.mean(points, axis=0)
    rel_points = points - centroid
    r_array = np.linalg.norm(rel_points, axis=1)
    theta_array = (np.arctan2(rel_points[:, 1], rel_points[:, 0]) + 2 * np.pi) % (2 * np.pi)

    r_array_n = r_array / (r_array.max() + 1e-8)

    r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    theta_bin_edges = np.linspace(0, 2 * np.pi, nbins_theta + 1)

    hist = np.zeros((nbins_r, nbins_theta))

    for r, theta in zip(r_array_n, theta_array):
        r_bin = np.searchsorted(r_bin_edges, r)
        theta_bin = np.searchsorted(theta_bin_edges, theta) - 1
        if 0 <= r_bin < nbins_r and 0 <= theta_bin < nbins_theta:
            hist[r_bin, theta_bin] += 1
    hist /= (np.linalg.norm(hist) + 1e-8)
    return hist.flatten()


def compute_sc_descriptors(contours):
    descriptors = []
    for cnt in contours:
        cnt_pts = cnt[:, 0, :]
        desc = shape_context_descriptor(cnt_pts)
        if desc is not None:
            descriptors.append(desc)
        else:
            # 轮廓点太少，返回零向量，避免匹配时出错
            descriptors.append(np.zeros(5*12))
    return np.array(descriptors)


def bidirectional_match(desc1, desc2, threshold=0.3):
    dist = cdist(desc1, desc2, metric='euclidean')
    matches = []
    for i in range(len(desc1)):
        j = np.argmin(dist[i])
        i_back = np.argmin(dist[:, j])
        if i_back == i and dist[i, j] < threshold:
            matches.append((i, j, dist[i, j]))
    return matches


# def draw_contour_matches(img1, img2, contours1, contours2, matches):
#     h1, w1 = img1.shape
#     h2, w2 = img2.shape
#     out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     out_img[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     out_img[:h2, w1:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

#     for (i1, i2, _) in matches:
#         cnt1 = contours1[i1]
#         cnt2 = contours2[i2]
#         M1 = cv2.moments(cnt1)
#         c1 = (int(M1['m10'] / (M1['m00']+1e-8)), int(M1['m01'] / (M1['m00']+1e-8)))
#         M2 = cv2.moments(cnt2)
#         c2 = (int(M2['m10'] / (M2['m00']+1e-8)) + w1, int(M2['m01'] / (M2['m00']+1e-8)))

#         cv2.circle(out_img, c1, 5, (0,255,255), 2)
#         cv2.circle(out_img, c2, 5, (0,255,255), 2)
#         cv2.line(out_img, c1, c2, (0,255,255), 1)

#     cv2.imshow('Shape Context Matches', out_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
def draw_contour_matches(img1, img2, contours1, contours2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    out_img[:h2, w1:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 随机生成匹配对颜色，保持颜色一致性
    colors = []
    random.seed(42)
    for _ in matches:
        colors.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

    for idx, (i1, i2, _) in enumerate(matches):
        color = colors[idx]
        cnt1 = contours1[i1]
        cnt2 = contours2[i2]

        # 画轮廓：第一个图的轮廓直接画
        cv2.drawContours(out_img, [cnt1], -1, color, 2)
        # 第二个图的轮廓需要偏移x坐标
        cnt2_shifted = cnt2.copy()
        cnt2_shifted[:, 0, 0] += w1
        cv2.drawContours(out_img, [cnt2_shifted], -1, color, 2)

        # 计算中心点
        M1 = cv2.moments(cnt1)
        c1 = (int(M1['m10'] / (M1['m00']+1e-8)), int(M1['m01'] / (M1['m00']+1e-8)))
        M2 = cv2.moments(cnt2)
        c2 = (int(M2['m10'] / (M2['m00']+1e-8)) + w1, int(M2['m01'] / (M2['m00']+1e-8)))

        # 画中心点和连线
        cv2.circle(out_img, c1, 5, color, -1)
        cv2.circle(out_img, c2, 5, color, -1)
        cv2.line(out_img, c1, c2, color, 1)
    cv2.imshow('Shape Context Matches', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img1 = cv2.imread('model_embeddings\model_img\moban_5.jpg', 0)
    # M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 60, 0.2)
    # img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    # img1 = cv2.imread('org_muban\muban_4.bmp', 0)
    # img1_copy = img1.copy()
    img2 = cv2.imread('mubiao_2.bmp', 0)
    cv2.namedWindow('Shape Context Matches', cv2.WINDOW_NORMAL)
    cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    # img2_copy = img2.copy()
    # for i in range(100):
    #     start_time = time.time()
    #     contours1 = extract_contours(img1)
    #     contours2 = extract_contours(img2)
    #     print(f"Extracting contours took {time.time() - start_time:.6f} seconds")
    #     desc1 = compute_sc_descriptors(contours1)
    #     desc2 = compute_sc_descriptors(contours2)

    #     matches = bidirectional_match(desc1, desc2, threshold=0.5)
    #     print(f"Matching took {time.time() - start_time:.6f} seconds")
    start_time = time.time()
    contours1 = extract_contours(img1)
    contours2 = extract_contours(img2)
    print(f"Extracting contours took {time.time() - start_time:.6f} seconds")
    desc1 = compute_sc_descriptors(contours1)
    desc2 = compute_sc_descriptors(contours2)

    matches = bidirectional_match(desc1, desc2, threshold=0.5)
    print(f"Matching took {time.time() - start_time:.6f} seconds")
    print(f"Found {len(matches)} matches")

    draw_contour_matches(img1, img2, contours1, contours2, matches)


if __name__ == '__main__':
    main()
