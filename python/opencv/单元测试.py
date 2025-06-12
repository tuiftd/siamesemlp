from Template_class import Template
import cv2
import numpy as np
def make_divisible(size, divisor=16, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(size + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * size:
        new_ch += divisor
    return new_ch

def resized(Image, divisor=16, min_ch=None,center:tuple=None):
    goal_size = make_divisible(size=min(Image.shape[0], Image.shape[1]), divisor=divisor, min_ch=min_ch)
    Image_resized,center = letterbox(Image, target_size=goal_size,center=center)
    return Image_resized,center

def letterbox(image, target_size=640,center:tuple=None):
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
    if center is None:
        return canvas,(center)
    else:
        return canvas ,(center*scale + np.array([left, top]))


if __name__ == '__main__':
    img_path = r'org_muban\muban_5.bmp'
    img = cv2.imread(img_path)
    template = Template(img=img,img_name='moban_5')
    template.show()
    template.MaskMake(method="OTSU")
    print(template.crop_size)
    template.resize_to_Template(divisor=16)
    template.save()
    # tl,center = resized(template.img_crop, divisor=256,center=template.center)
    # print(center)
    # cv2.circle(tl, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
    # cv2.imshow("Template", tl)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



