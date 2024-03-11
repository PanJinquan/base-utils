# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-29 15:16:35
    @Brief  :
"""

import cv2
import numpy as np
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import corner_utils


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
    corners = corner_utils.get_order_points(contour)
    return corners


if __name__ == '__main__':
    image_dir = "../data/mask"
    files = file_utils.get_files_lists(image_dir)
    for file in files:
        image = image_utils.read_image(file, size=(250, 250))
        mask = image_utils.get_image_mask(image, inv=False)
        corners = get_minboundquad(mask, vis=True)
        image = image_utils.draw_image_points_lines(image, corners, fontScale=1.0, thickness=2)
        image_utils.cv_show_image("winname", image, use_rgb=False)
