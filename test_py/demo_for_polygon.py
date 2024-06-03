# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-29 15:16:35
    @Brief  :
"""

import cv2
import numpy as np
from pybaseutils import image_utils, file_utils, coords_utils
from pybaseutils.cvutils import corner_utils


# 多边形周长
# shape of polygon: [N, 2]
def Perimeter(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    permeter = 0.
    for i in range(N):
        permeter += np.linalg.norm(polygon[i - 1] - polygon[i])
    return permeter


# 面积
def Area(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    area = 0.
    vector_1 = polygon[1] - polygon[0]
    for i in range(2, N):
        vector_2 = polygon[i] - polygon[0]
        area += np.abs(np.cross(vector_1, vector_2))
        vector_1 = vector_2
    return area / 2


# |r| < 1
# r > 0, 内缩
# r < 0, 外扩
def calc_shrink_width(polygon: np.array, r):
    area = Area(polygon)
    perimeter = Perimeter(polygon)
    L = area * (1 - r ** 2) / perimeter
    return L if r > 0 else -L


def shrink_polygon(polygon: np.array, r):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    shrinked_polygon = []
    L = calc_shrink_width(polygon, r)
    for i in range(N):
        Pi = polygon[i]
        v1 = polygon[i - 1] - Pi
        v2 = polygon[(i + 1) % N] - Pi

        normalize_v1 = v1 / np.linalg.norm(v1)
        normalize_v2 = v2 / np.linalg.norm(v2)

        sin_theta = np.abs(np.cross(normalize_v1, normalize_v2))

        Qi = Pi + L / sin_theta * (normalize_v1 + normalize_v2)
        shrinked_polygon.append(Qi)
    return np.asarray(shrinked_polygon)


def get_minboundquad(mask, n_corners=4, max_iter=10, vis=False):
    """
    获得最小包络四边形
    https://blog.csdn.net/Robin__Chou/article/details/112705540
    :param mask:
    :param ksize:
    :param blur:
    :param vis:
    :return:
    """
    mask = image_utils.get_mask_morphology(mask, ksize=5, binarize=False, op=cv2.MORPH_OPEN, itera=2)
    # 获得所有轮廓
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return np.zeros((0, 2), np.int32)
    # Keeping only the largest detected contour.
    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]
    contour = contour.reshape(-1, 2)
    # contour = corner_utils.myapproxPolyDP(contour, n_corners, max_iter=100, lr=0.1, log=True)
    # Sorting the corners and converting them to desired shape.
    # corners = corner_utils.get_order_points(contour)
    return contour


if __name__ == '__main__':
    image_dir = "../data/mask"
    files = file_utils.get_files_lists(image_dir)
    for file in files:
        image = image_utils.read_image(file, size=(250, 250))
        mask = image_utils.get_image_mask(image, inv=False)
        corners = get_minboundquad(mask, vis=True)
        # corners = shrink_polygon(corners, -0.5)
        corners = coords_utils.shrink_polygon_pyclipper(corners, 1.2)
        image = image_utils.draw_image_contours(image, [corners], thickness=2)
        image_utils.cv_show_image("winname", image, use_rgb=False)
