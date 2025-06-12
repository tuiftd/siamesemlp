import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt


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


def polar_histogram(points, center=None, radius_bins=6, angle_bins=12):
    if len(points) < 5:
        return None
    if center is None:
        center = np.mean(points, axis=0)
    relative = points - center
    r = np.linalg.norm(relative, axis=1)
    theta = (np.arctan2(relative[:, 1], relative[:, 0]) + 2 * np.pi) % (2 * np.pi)
    r_norm = r / (r.max() + 1e-8)
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


def extract_descriptors(img, stride=4, window=32):
    h, w = img.shape
    descriptors = []
    positions = []
    for y in range(0, h - window, stride):
        for x in range(0, w - window, stride):
            patch = img[y:y+window, x:x+window]
            contour_pts = extract_contour_points(patch)
            if len(contour_pts) == 0:
                continue
            contour_pts = contour_pts + np.array([x, y])
            center = np.mean(contour_pts, axis=0)
            desc = polar_histogram(contour_pts, center)
            if desc is not None:
                descriptors.append(desc)
                positions.append(center)
    return np.array(descriptors), np.array(positions)


def bidirectional_match(desc1, desc2, pos1, pos2, threshold=0.3):
    """欧氏距离的双向匹配"""
    dist_matrix = cdist(desc1, desc2, 'euclidean')
    matches = []
    for i in range(len(desc1)):
        j = np.argmin(dist_matrix[i])
        i_back = np.argmin(dist_matrix[:, j])
        if i_back == i and dist_matrix[i, j] < threshold:
            matches.append((pos1[i], pos2[j], dist_matrix[i, j]))
    return matches


def ransac_affine_verification(matches):
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
    for p1, p2, _ in matches:
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        cv2.line(result, p1, p2, (0, 255, 0), 1)
        cv2.circle(result, p1, 2, (255, 0, 0), -1)
        cv2.circle(result, p2, 2, (255, 0, 0), -1)
    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img1 = cv2.imread('org_muban/muban_2.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1] // 2, img1.shape[0] // 2), 60, 0.8)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

    print("Extracting descriptors from target...")
    desc1, pos1 = extract_descriptors(img1)
    print("Extracting descriptors from template...")
    desc2, pos2 = extract_descriptors(img2)

    print(f"Target features: {len(desc1)}, Template features: {len(desc2)}")

    raw_matches = bidirectional_match(desc1, desc2, pos1, pos2, threshold=0.5)
    verified_matches, _ = ransac_affine_verification(raw_matches)

    print(f"Verified matches: {len(verified_matches)}")
    draw_matches(img1, img2, verified_matches)


if __name__ == "__main__":
    main()
