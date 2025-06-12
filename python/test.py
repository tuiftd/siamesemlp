
import numpy as np

def hu_distance_i1(hu_a, hu_b):
    """
    CONTOURS_MATCH_I1 距离计算
    公式：sum(|1/hu_b - 1/hu_a|)
    """
    inv_hu_a = 1.0 / (hu_a + 1e-10)  # 加小常数防止除零
    inv_hu_b = 1.0 / (hu_b + 1e-10)
    return np.sum(np.abs(inv_hu_b - inv_hu_a))

def hu_distance_i2(hu_a, hu_b):
    """
    CONTOURS_MATCH_I2 距离计算
    公式：sum(|hu_b - hu_a|)
    """
    return np.sum(np.abs(hu_b - hu_a))

def hu_distance_i3(hu_a, hu_b):
    """
    CONTOURS_MATCH_I3 距离计算
    公式：sum(|hu_b - hu_a| / |hu_a|)
    """
    diff = np.abs(hu_b - hu_a)
    norm = np.abs(hu_a) + 1e-10  # 加小常数防止除零
    return np.sum(diff / norm)

# 示例使用 --------------------------------------------------
if __name__ == "__main__":
    # 假设已经计算并取对数的Hu矩（示例数据）
    hu_template = np.array([-3.2, -5.1, -8.4, -9.2, -12.5, -15.1, -18.3])  # 模板Hu矩
    hu_query = np.array([-3.1, -5.3, -8.1, -9.5, -12.2, -15.3, -18.1])     # 查询Hu矩

    # 计算三种距离
    dist_i1 = hu_distance_i1(hu_template, hu_query)
    dist_i2 = hu_distance_i2(hu_template, hu_query)
    dist_i3 = hu_distance_i3(hu_template, hu_query)

    print(f"I1距离: {dist_i1:.6f}")
    print(f"I2距离: {dist_i2:.6f}")
    print(f"I3距离: {dist_i3:.6f}")

    # 与OpenCV的matchShapes结果对比验证
    import cv2

    # 创建两个测试轮廓（这里用简单矩形模拟）
    contour1 = np.array([[[0,0]], [[10,0]], [[10,10]], [[0,10]]], dtype=np.int32)
    contour2 = np.array([[[0,0]], [[4,0]], [[9,9]], [[0,7]]], dtype=np.int32)

    # OpenCV计算结果
    cv_i1 = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
    cv_i2 = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0)
    cv_i3 = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I3, 0)

    print("\nOpenCV验证结果:")
    print(f"CV I1: {cv_i1:.6f}")
    print(f"CV I2: {cv_i2:.6f}")
    print(f"CV I3: {cv_i3:.6f}")