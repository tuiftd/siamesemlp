import cv2
import numpy as np
import os
def make_divisible(size, divisor=16, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(size + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * size:
        new_ch += divisor
    return new_ch

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
        canvas = np.full((target_size, target_size), 0, dtype=np.uint8)  # 114是YOLO的填充灰度值
    else:
        canvas = np.full((target_size, target_size,3), 0, dtype=np.uint8)  # 114是YOLO的填充灰度值
    # 将缩放后的图像居中放置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

def adaptiveThreshold(image, block_size=11, C=2, maxval=255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,trehshold_type=cv2.THRESH_BINARY):
    return cv2.adaptiveThreshold(image, maxValue=maxval, adaptiveMethod=adaptive_method, thresholdType = trehshold_type, blockSize= block_size, C = C)
def OTSU(image,trehshold_type:int=cv2.THRESH_BINARY):
    return cv2.threshold(image, 0, 255, trehshold_type + cv2.THRESH_OTSU)[1]
def threshold(image, threshold_value:int=127, maxval:int=255, trehshold_type:int=cv2.THRESH_BINARY):
    return cv2.threshold(image, thresh = threshold_value, maxval = maxval, type = trehshold_type)[1]


class Template:
    """除了灰度,都是3通道的,无论输入的是什么图像"""
    count = 0
    def __init__(self, img_path:str = None,img:np.ndarray=None,img_name:str=None):
        if img is None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img
        if self.img is None:
            raise ValueError("Image not found")
        if len(self.img.shape)!= 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        if img_name is None:
            self.name = "template"+str(Template.count)
        else:
            self.name = img_name
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.RGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w, self.c = self.img.shape
        Template.count += 1
        if img_path is not None:
            self.img_path = os.path.abspath(img_path)
        else:
            self.img_path = None
    def show(self):
        letterbox_img = letterbox(self.img)
        cv2.imshow("Template", letterbox_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def __del__(self):
        Template.count -= 1
    def save(self, save_path:str = None):
        if save_path is None:
            if self.img_path is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            else:   
                base_dir = os.path.dirname(self.img_path)
            save_path = os.path.join(base_dir, "Template",self.name+".jpg")
        else:
            save_path = os.path.join(save_path, "Template",self.name+".jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imshow("Template", self.img_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        success =   cv2.imwrite(save_path, self.img_crop)
        if success:
            print(f"Template saved to {save_path}")
        else:
            print(f"Failed to save Template to {save_path}")
    def MaskMake(self,
                 method:str="adaptiveThreshold",
                 brightObject:bool=True,
                 **kwargs:dict
                 ):
        """生成图像处理的二值化掩码(mask)。
    
    支持三种二值化方法：
    - `adaptiveThreshold`: 自适应阈值（默认）
    - `OTSU`: 大津算法（自动阈值）
    - `threshold`: 固定阈值

    Args:
        method (str): 二值化方法，可选 `adaptiveThreshold`、`OTSU` 或 `threshold`。
        brightObject (bool): 若为 `True`，目标物体为亮色（背景为暗）；否则反转。
        ​**kwargs: 传递给具体二值化方法的参数，例如：
            - adaptiveThreshold(image, block_size=11, C=2, maxval=255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,trehshold_type=cv2.THRESH_BINARY):
            - OTSU(image,trehshold_type:int=cv2.THRESH_BINARY):
            - threshold(image, threshold_value:int=127, maxval:int=255, threshold_type:int=cv2.THRESH_BINARY):

    Returns:
        np.ndarray: 二值化后的掩码图像(0 和 255)。

    Examples:
        >>> mask = MaskMake(method="OTSU",**kwargs)
        >>> cv2.imshow("Mask", mask)

    Notes:
        具体参数详见 OpenCV 文档：
        - `cv2.adaptiveThreshold`
        - `cv2.threshold`（含 OTSU 模式）
    """
        method_dict = {"adaptiveThreshold": adaptiveThreshold,"OTSU": OTSU,"threshold": threshold}
        if method not in method_dict:
            raise ValueError("method must be adaptiveThreshold, OTSU or threshold")
        if brightObject:
            threshold_type = cv2.THRESH_BINARY
        else:
            threshold_type = cv2.THRESH_BINARY_INV
        kwargs["trehshold_type"] = threshold_type
        method_func = method_dict[method]
        Binary_Image = method_func(self.gray, **kwargs)
        contours,_ = cv2.findContours(Binary_Image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img_h,img_w = Binary_Image.shape[:2]
        print(img_w,img_h)
        mask = np.zeros(Binary_Image.shape[:2], np.uint8)
        for contour in contours:
          #获取轮廓的面积
            if len(contour) < 10:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if w/img_w > 0.90 and h/img_h > 0.90: #防止图片边框被检索到
                continue
            area = cv2.contourArea(contour)
            # 面积小于1000的轮廓忽略
            if area < 1000:
                continue
            cv2.fillPoly(mask, pts=[contour], color=255)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        x,y,w,h = cv2.boundingRect(mask)
        Template_only = cv2.bitwise_and(self.img,self.img,mask=mask)
        for contour in contours:
            #获取轮廓的面积
            if len(contour) < 10:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if w/img_w > 0.90 and h/img_h > 0.90:
                continue
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            # 计算轮廓的矩
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            centers.append(center)
        #求平均中心点
        if len(centers) > 0:
            center = np.mean(centers,axis=0)
            print(center)
        img_crop = Template_only[y:y+h,x:x+w]
        center = center-np.array([x,y])
        self.center = center
        self.img_crop =img_crop
        self.img_mask = mask
        self.crop_size = [w,h]
        img_crop = letterbox(img_crop)
        mask = letterbox(mask)
        if len(img_crop.shape) == 3:
            mask_copy = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = letterbox(self.img)
        h_stack = cv2.hconcat([img_crop, mask_copy, img])
        cv2.imshow("Mask", h_stack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask,self.center,self.img_crop
    def resize_to_Template(self, target_size:int=None, divisor=16):
        """调整图像到最符合倍率，最接近目标尺寸。"""
        if target_size is None:
            target_size = min(self.img_crop.shape[0], self.img_crop.shape[1])
        else:
            target_size = target_size
        goal_size = make_divisible(target_size, divisor)
        h, w = self.img_crop.shape[:2]
        scale = min(goal_size / h, goal_size / w)
    # 等比例缩放后的新尺寸
        new_h, new_w = int(h * scale), int(w * scale)
        # 缩放图像
        resized = cv2.resize(self.img_crop, (new_w, new_h))
        # 创建目标正方形画布
        canvas = np.full((goal_size, goal_size,3), 0, dtype=np.uint8)  # 114是YOLO的填充灰度值
        # 将缩放后的图像居中放置
        top = (goal_size - new_h) // 2
        left = (goal_size - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized
        self.img_crop = canvas
        self.center = self.center*scale+np.array([left,top])
        self.crop_size = [new_w,new_h]
        # cv2.circle(canvas, (int(self.center[0]), int(self.center[1])), 5, (0, 0, 255), -1)
        cv2.imshow("Template", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    

        
    
        