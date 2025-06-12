import cv2
import numpy as np
from PIL import Image

image_path =r'date\MPEG7_dataset\train\bell\bell-2.gif'
img = Image.open(image_path).convert('L')
img_np = np.array(img)
contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(img_np)
cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
canny = cv2.Canny(img_np, 100, 200)
# canny = cv2.blur(canny, (5,5))
cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()