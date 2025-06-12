import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def extract_contours(img, min_area=1):
    edges = cv2.Canny(img, 50, 150)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilate = cv2.dilate(edges, kernel_dilate)
    img_erode = cv2.erode(img_dilate, kernel_erode)
    contours, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered


def fourier_descriptor(contour, n_descriptors=32):
    # contour shape: (N,1,2)
    contour = contour[:, 0, :]
    complex_pts = contour[:, 0] + 1j * contour[:, 1]
    # Fourier transform
    fourier_res = np.fft.fft(complex_pts)
    # Use only low frequency descriptors (centralized)
    descriptors = np.concatenate([fourier_res[:n_descriptors//2],
                                  fourier_res[-n_descriptors//2:]])
    # Normalize for scale invariance
    descriptors /= np.abs(descriptors[1]) + 1e-8
    # Return real + imag parts concatenated (vector of length n_descriptors*2)
    desc = np.concatenate([descriptors.real, descriptors.imag])
    return desc


def bidirectional_match(desc1, desc2, threshold=1):
    dist = cdist(desc1, desc2, metric='euclidean')
    matches = []
    for i in range(len(desc1)):
        j = np.argmin(dist[i])
        i_back = np.argmin(dist[:, j])
        if i_back == i and dist[i, j] < threshold:
            matches.append((i, j, dist[i, j]))
    return matches


def draw_contour_matches(img1, img2, contours1, contours2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    out_img[:h2, w1:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for (i1, i2, _) in matches:
        cnt1 = contours1[i1]
        cnt2 = contours2[i2]
        # Compute centroids
        M1 = cv2.moments(cnt1)
        c1 = (int(M1['m10'] / (M1['m00']+1e-8)), int(M1['m01'] / (M1['m00']+1e-8)))
        M2 = cv2.moments(cnt2)
        c2 = (int(M2['m10'] / (M2['m00']+1e-8)) + w1, int(M2['m01'] / (M2['m00']+1e-8)))

        cv2.circle(out_img, c1, 5, (0,255,0), 2)
        cv2.circle(out_img, c2, 5, (0,255,0), 2)
        cv2.line(out_img, c1, c2, (0,255,0), 1)

    cv2.imshow('Fourier Descriptor Matches', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img1 = cv2.imread('org_muban/muban_2.bmp', 0)
    M = cv2.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 60, 0.8)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

    contours1 = extract_contours(img1)
    contours2 = extract_contours(img2)

    desc1 = [fourier_descriptor(c) for c in contours1]
    desc2 = [fourier_descriptor(c) for c in contours2]

    matches = bidirectional_match(np.array(desc1), np.array(desc2), threshold=0.7)
    print(f"Found {len(matches)} matches")

    draw_contour_matches(img1, img2, contours1, contours2, matches)


if __name__ == '__main__':
    main()
