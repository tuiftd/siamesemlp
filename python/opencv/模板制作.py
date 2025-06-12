
import cv2
import numpy as np
print(cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
def letterbox(image, target_size=640):
    # 原始图像尺寸
    h, w = image.shape[:2]
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    # 等比例缩放后的新尺寸
    new_h, new_w = int(h * scale), int(w * scale)
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    # 创建目标正方形画布
    if len(resized.shape) == 2:
        canvas = np.full((target_size, target_size), 114, dtype=np.uint8)  # 114是YOLO的填充灰度值
    else:
        canvas = np.full((target_size, target_size,3), 114, dtype=np.uint8)  # 114是YOLO的填充灰度值
    # 将缩放后的图像居中放置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas
img = cv2.imread('mobal.bmp',cv2.IMREAD_COLOR_BGR)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_copy = img.copy()
#自适应区域阈值二值化
thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
thresh_boxed = letterbox(thresh,640)
cv2.imshow('thresh',thresh_boxed)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img_h,img_w = thresh.shape[:2]
print(img_w,img_h)
mask = np.zeros(img.shape[:2], np.uint8)
for contour in contours:
    #获取轮廓的面积
    if len(contour) < 3:
        continue
    area = cv2.contourArea(contour)
    #返回轮廓的最小外接矩形
    x,y,w,h = cv2.boundingRect(contour)
    if w/img_w > 0.90 and h/img_h > 0.90:
        continue
    # 绘制矩形
    # 面积小于1000的轮廓忽略
    if area < 1000:
        continue
    # 绘制轮廓
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
    #填充区域
    cv2.fillPoly(mask, pts=[contour], color=255)
    #cv2.fillPoly(img_copy, pts=[contour], color=(0, 255, 0))
    # 计算轮廓的矩
# thresh = cv2.
cv2.imshow('img',mask)
cv2.waitKey(0)  
cv2.destroyAllWindows()
contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
centers = []
x,y,w,h = cv2.boundingRect(mask)
img_copy = cv2.bitwise_and(img_copy,img_copy,mask=mask)
#cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
for contour in contours:
    #获取轮廓的面积
    if len(contour) < 3:
        continue
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    if w/img_w > 0.90 and h/img_h > 0.90:
        continue
    # 绘制矩形
    # 面积小于1000的轮廓忽略
    if area < 1000:
        continue
    # 计算轮廓的矩
    rect = cv2.minAreaRect(contour)
    center = rect[0]
    centers.append(center)
    # 绘制矩形
#求平均中心点
if len(centers) > 0:
    center = np.mean(centers,axis=0)
    print(center)
    #cv2.circle(img_copy, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
cv2.imshow('img',img_copy)
cv2.waitKey(0)  
cv2.destroyAllWindows()
#裁剪图像
img_crop = img_copy[y:y+h,x:x+w]
center = center-np.array([x,y])
#cv2.circle(img_crop, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
cv2.imshow('img_crop',img_crop)
cv2.waitKey(0)  
cv2.destroyAllWindows()
#硬阈值二值化
ret,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(cv2.THRESH_BINARY+cv2.THRESH_OTSU)
