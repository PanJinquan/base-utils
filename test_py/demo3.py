# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-08 14:10:15
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils


def process_image_border(image, scale=(0.96, 0.96), vis=True):
    """
    对原图边界进行修复
    :param image: BGR原始图像
    :param scale: 缩放大小
    :param vis: 可视化效果
    :return: dest_： 返回边界修复后的原图
             gray2： 返回边界修复后的灰度图
    """
    h, w = image.shape[:2]
    gray1 = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask1 = image_utils.get_image_mask(gray1, inv=False)
    point = image_utils.find_mask_contours(mask1, max_nums=1, mode=cv2.RETR_EXTERNAL)
    boxes = image_utils.points2boxes(point)
    boxes = image_utils.extend_xyxy(boxes, scale=scale)
    box2 = boxes[0]
    # box2 = [50, 50, 200, 200]
    crop_ = image_utils.get_box_crop(image, box2)
    dest_ = image_utils.get_box_crop_recover(crop_, box2, size=(w, h), borderType=cv2.BORDER_REPLICATE)
    gray2 = 255 - cv2.cvtColor(dest_, cv2.COLOR_BGR2GRAY)
    if vis:
        v1 = image_utils.image_hstack([image, gray1], split_line=True)
        v2 = image_utils.image_hstack([dest_, gray2], split_line=True)
        res = image_utils.image_vstack([v1, v2], split_line=True)
        image_utils.show_image("result", res)
    return dest_, gray2


def test_image_dir(image_dir, vis=True):
    """
    :param image_dir: 图片文件或图片文件夹
    :param vis:
    :return:
    """
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        image = cv2.imread(image_file)
        image = image_utils.resize_image(image, size=(200, 200))
        dest, gray = process_image_border(image, vis=vis)


if __name__ == '__main__':
    """pip install --upgrade pybaseutils -i https://pypi.org/simple"""
    image_dir = "/home/PKing/Downloads/转灰度/20250708-095351.jpg"
    test_image_dir(image_dir)
