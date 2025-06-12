import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
import cv2

def compute_log_hu_moments(image_path):
    """读取图像，计算 Hu 矩（log 变换）"""
    

# 用 PIL 读取 gif 并转换为灰度 numpy 数组
    img_pil = Image.open(image_path).convert('L')  # L 模式为灰度
    img = np.array(img_pil)
    # img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    # 二值化处理
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 获取普通矩 + Hu 矩
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)

    # log 缩放以便可视化
    log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return log_hu.flatten()

def plot_hu_radar(log_hu, label='Image'):
    """绘制雷达图展示 Hu 矩特征"""
    angles = [n / 7 * 2 * pi for n in range(7)]
    angles += angles[:1]  # 闭合雷达环
    values = np.append(log_hu, log_hu[0])

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], [f'Hu{i+1}' for i in range(7)])
    plt.title(f'Hu Moments Radar Plot: {label}')
    plt.legend(loc='upper right')
    plt.show()

# === 示例用法 ===
if __name__ == "__main__":
    image_path = r"date\MPEG7\beetle-9.gif"  
    log_hu = compute_log_hu_moments(image_path)
    plot_hu_radar(log_hu, label=image_path)
