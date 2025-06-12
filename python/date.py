import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# === 可调参数 ===
NUM_BASE_IMAGES = 50
AUG_PER_IMAGE = 20
TARGET_SIZE = 256
OUTPUT_DIR = Path("dataset")
TRAIN_RATIO = 0.7

# === 几何参数 ===
CONCAVE_POLY_PROB = 0.5
ARC_EDGE_PROB = 0.3

# === 生成凹/凸多边形或含弧形 ===
def generate_random_polygon(size=(256, 256), concave_prob=0.5, arc_prob=0.3):
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)
    w, h = size

    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(h // 4, 3 * h // 4)
    radius = random.randint(30, 80)
    num_points = random.randint(5, 9)

    angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_points)])
    points = []
    for a in angles:
        r_mod = radius * random.uniform(0.6, 1.2 if random.random() < concave_prob else 1.0)
        x = int(center_x + r_mod * np.cos(a))
        y = int(center_y + r_mod * np.sin(a))
        points.append((x, y))

    if random.random() < arc_prob:
        draw.line(points + [points[0]], fill=255, width=2)
        for i in range(len(points)):
            bbox = [
                points[i][0] - 5, points[i][1] - 5,
                points[i][0] + 5, points[i][1] + 5
            ]
            draw.pieslice(bbox, 0, 360, fill=255)
    else:
        draw.polygon(points, fill=255)

    return img

# === letterbox 缩放函数 ===
def letterbox_gray(image, target_size=256):
    w, h = image.size
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('L', (target_size, target_size), 0)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas

# === 随机变换：旋转 + 平移 + 缩放 ===
def random_transform(img, target_size=256):
    rows, cols = img.shape

    # 随机仿射变换
    angle = random.uniform(-45, 45)
    tx = random.uniform(-0.2 * cols, 0.2 * cols)
    ty = random.uniform(-0.2 * rows, 0.2 * rows)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=0)

    # 缩放（保持比例）
    scale_factor = random.uniform(0.7, 1.2)
    new_w = int(cols * scale_factor)
    new_h = int(rows * scale_factor)
    scaled = cv2.resize(transformed, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # letterbox 到 256×256
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = max((target_size - new_w) // 2, 0)
    y_offset = max((target_size - new_h) // 2, 0)
    x_end = x_offset + min(new_w, target_size)
    y_end = y_offset + min(new_h, target_size)
    crop_w = min(new_w, target_size)
    crop_h = min(new_h, target_size)
    canvas[y_offset:y_end, x_offset:x_end] = scaled[:crop_h, :crop_w]
    return canvas

# === 创建图像增强数据集 ===
def generate_dataset():
    for i in range(NUM_BASE_IMAGES):
        base_img = generate_random_polygon((TARGET_SIZE, TARGET_SIZE), CONCAVE_POLY_PROB, ARC_EDGE_PROB)
        base_pil = letterbox_gray(base_img)
        aug_images = []

        for aug_id in range(AUG_PER_IMAGE):
            aug_img = random_transform(np.array(base_pil))
            aug_images.append(aug_img)

        # 划分每张基础图像的20个增强样本为训练 / 测试
        indices = list(range(AUG_PER_IMAGE))
        train_idx, test_idx = train_test_split(indices, train_size=TRAIN_RATIO, shuffle=True)

        for split, id_list in zip(['train', 'test'], [train_idx, test_idx]):
            sample_dir = OUTPUT_DIR / split / f"{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for idx in id_list:
                Image.fromarray(aug_images[idx]).save(sample_dir / f"{idx:02d}.png")

if __name__ == "__main__":
    generate_dataset()
    print("✅ 数据集生成完成（每张基础图像按比例划分训练 / 测试）")
