# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np


def get_joints_heatmap(joints, input_size, out_size, sigma=2):
    """
    生成关键点热力图，每个关键点的热力图是一个二维高斯分布，关键点的可见性为1时，热力图值为1，否则为0
    :param joints:  shape is (num_joints, 3), (x, y, v) where joints[:, 2] is the visibility of each joint
    :param input_size: shape is (width,height)
    :param out_size: shape is (width,height)
    :return:
    """
    num_joints = int(len(joints))
    weights = np.ones((num_joints), dtype=np.float32)
    heatmap = []
    tmp_size = sigma * 3
    for id in range(num_joints):
        hm = np.zeros((out_size[1], out_size[0]), dtype=np.float32)
        y_r = input_size[1] / out_size[1]
        x_r = input_size[0] / out_size[0]
        # feat_stride = image_size / self.heatmap_size
        mu_x = int(joints[id][0] / x_r + 0.5)  # y
        mu_y = int(joints[id][1] / y_r + 0.5)  # x
        x, y = joints[id][0], joints[id][1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= out_size[0] or ul[1] >= out_size[1] or br[0] < 0 or br[1] < 0:
            weights[id] = 0
        elif x < 0 or x > input_size[0] or y < 0 or y > input_size[1]:
            weights[id] = 0
        else:
            weights[id] = 1.0
            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], out_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], out_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], out_size[0])
            img_y = max(0, ul[1]), min(br[1], out_size[1])
            hm[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                hm[img_y[0]:img_y[1], img_x[0]:img_x[1]],
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        heatmap.append(hm)
    heatmap = np.asarray(heatmap)
    return heatmap, weights


def get_points_heatmap(points, size, sigma=2):
    """
    生成关键点热力图,多个关键点共享一个热力图
    :param points: (num_joints, 2): 关键点列表，每个元素为 (x, y)，注意x对应width，y对应height
    :param size: (width, height) 输出图像尺寸
    :param sigma (float): 高斯分布的标准差，控制热力图扩散范围
    :return: heatmaps: numpy array, shape = (height, width)
    """
    width, height = size
    heatmap = np.zeros((height, width), dtype=np.float32)
    # 为每个关键点生成一个热力图
    for pts in points:
        x, y = pts[0], pts[1]
        if x < 0 or y < 0 or x >= width or y >= height:
            # 如果关键点在图像外，跳过（或可置为全0）
            continue
        # 创建坐标网格
        y_grid, x_grid = np.ogrid[:height, :width]
        # 计算每个像素到关键点的欧氏距离的平方
        dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
        dist_sq = dist_sq.astype(np.float32)
        # 生成2D高斯热力图
        hp = np.exp(-dist_sq / (2 * sigma ** 2))
        heatmap = np.max([heatmap, hp], axis=0)
    return heatmap


def get_curves_heatmap(points, size, sigma=5, thickness=1):
    """
    使用 OpenCV绘制曲线并生成热力图
    :param points: (N,nums, 2) array or list of [x, y] points
    :param size: (width, height) 输出图像尺寸
    :param sigma: 高斯热力图标准差
    :param thickness: 绘制曲线的线宽（建议=1）
    :return: heatmap: (height, width) float32 热力图
    """
    width, height, = size
    # 1. 创建空白二值掩码（全黑）
    mask = np.zeros((height, width), dtype=np.uint8)
    # 2. 将曲线点转换为 int32 并绘制连线
    for pts in points:
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    # 3. 计算每个像素到最近白色像素（曲线）的距离,使用 cv2.distanceTransform 计算 L2 距离（欧氏距离）
    dist = cv2.distanceTransform(255 - mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    # 4. 转换为高斯热力图: exp(-dist^2 / (2*sigma^2))
    heatmap = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    heatmap = heatmap.astype(np.float32)
    return heatmap


def get_heatmap_points_max(heatmap):
    """
    直接取热力图中最大值的坐标（整像素精度）
    返回 (x, y)
    """
    h, w = heatmap.shape
    idx = np.argmax(heatmap)
    y = idx // w
    x = idx % w
    center = (x, y)
    scores = heatmap[y, x]
    center, scores = np.asarray([center], dtype=np.int32), np.asarray([scores], dtype=np.float32)
    return center, scores


def get_heatmap_points_gauss(heatmap, threshold=0.0):
    """
    从单通道热力图中提取关键点坐标（带亚像素 refinement）
    对最大值周围的 3×3 区域进行二次插值
    参数:
        heatmap: (H, W) array, 热力图响应
        threshold: 最小置信度阈值，低于则返回 None
    返回:
        (x, y, score) 或 None（如果最大值低于阈值）
    """
    H, W = heatmap.shape
    # 找到最大值位置
    idx = np.argmax(heatmap)
    y_max = idx // W
    x_max = idx % W
    scores = heatmap[y_max, x_max]
    center, scores = (-1, -1), float(scores)
    if scores < threshold:
        center, scores = np.asarray([center], dtype=np.float32), np.asarray([scores], dtype=np.float32)
        return center, scores  # 或返回 (0, 0, 0)
    # 如果最大值在边界，无法插值，直接返回
    if x_max == 0 or x_max == W - 1 or y_max == 0 or y_max == H - 1:
        center = (float(x_max), float(y_max))
        center, scores = np.asarray([center], dtype=np.float32), np.asarray([scores], dtype=np.float32)
        return center, scores
    # 获取 3x3 邻域
    patch = heatmap[y_max - 1:y_max + 2, x_max - 1:x_max + 2]
    # 计算 x 方向的偏移（使用左右像素）
    dx = 0.5 * (patch[1, 2] - patch[1, 0])
    dy = 0.5 * (patch[2, 1] - patch[0, 1])
    # 亚像素坐标
    x = x_max + dx
    y = y_max + dy
    center = (float(x), float(y))
    center, scores = np.asarray([center], dtype=np.float32), np.asarray([scores], dtype=np.float32)
    return center, scores


def get_heatmap_points_gauss_fit(heatmap):
    """
    使用高斯拟合计算中心点
    Args:
        heatmap: [H, W]
    Returns:
        center: (x, y)
    """
    H, W = heatmap.shape
    # 找到最大值位置作为初始估计
    max_loc = np.unravel_index(heatmap.argmax(), heatmap.shape)
    y0, x0 = max_loc
    # 定义局部区域（如5x5窗口）
    half_size = 2
    x_min = max(0, x0 - half_size)
    x_max = min(W, x0 + half_size + 1)
    y_min = max(0, y0 - half_size)
    y_max = min(H, y0 + half_size + 1)
    # 提取局部区域
    local_hm = heatmap[y_min:y_max, x_min:x_max]
    local_h, local_w = local_hm.shape
    if local_h == 0 or local_w == 0:
        scores = heatmap[int(x0), int(y0)]
        center, scores = np.asarray([[x0, y0]], dtype=np.float32), np.asarray([scores], dtype=np.float32)
        return center, scores
    # 创建局部坐标网格
    y_local, x_local = np.mgrid[0:local_h, 0:local_w]
    # 对局部区域进行高斯拟合（简化版本）
    # 实际应用中可以使用更精确的拟合方法
    weights = local_hm.flatten()
    x_flat = (x_local + x_min).flatten()
    y_flat = (y_local + y_min).flatten()
    if weights.sum() > 0:
        x_center = np.average(x_flat, weights=weights)
        y_center = np.average(y_flat, weights=weights)
    else:
        x_center, y_center = x0, y0
    center = (float(x_center), float(y_center))
    scores = heatmap[int(y_center), int(x_center)]
    center, scores = np.asarray([center], dtype=np.float32), np.asarray([scores], dtype=np.float32)
    return center, scores


def example01():
    """绘制直线热力图"""
    from pybaseutils import image_utils
    input_size = [256, 256]
    points = [[[50, 30], [180, 25], [230, 200], [60, 230]]]  # (N,nums, 2)
    heatmap = get_curves_heatmap(points, input_size)
    image_utils.cv_show_image("heatmap", heatmap, delay=0)


def example02():
    """绘制关键点热力图"""
    from pybaseutils import image_utils
    input_size = [256, 256]
    joints = [[50, 30, 1], [180, 25, 1], [230, 200, 1], [60, 230, 1]]  # (N,nums, 2)
    joints = [[50, 30], [180, 25], [230, 200], [60, 230]]  # (N,nums, 2)
    heatmap_size = [128, 128]
    joints = np.asarray(joints)
    heatmap, weight = get_joints_heatmap(joints, input_size, heatmap_size, sigma=2)
    heatmap = np.max(heatmap, axis=0)
    image_utils.cv_show_image("heatmap", heatmap, delay=0)


def example03():
    """绘制关键点热力图"""
    from pybaseutils import image_utils
    input_size = [256, 256]
    joints = [[50, 30], [52, 32]]  # (N,nums, 2)
    joints = np.asarray(joints)
    heatmap = get_points_heatmap(joints, input_size, sigma=5)
    heatmap = cv2.resize(heatmap, (640, 256))
    # center, scores = get_heatmap_points_gauss(heatmap)
    center, scores = get_heatmap_points_gauss_fit(heatmap)
    # center, scores = get_heatmap_points_max(heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    heatmap = image_utils.draw_points_texts(heatmap, center, texts=scores, color=(0, 255, 0), thickness=1)
    image_utils.cv_show_image("heatmap", heatmap, delay=0)


if __name__ == "__main__":
    # example01()
    # example02()
    example03()
