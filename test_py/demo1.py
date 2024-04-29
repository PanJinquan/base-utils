# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc


def test_transform(image_file):
    src = cv2.imread(image_file)
    mask = image_utils.get_image_mask(src)
    contours = image_utils.find_mask_contours(mask)
    src_pts = image_utils.find_minAreaRect(contours, order=True)[0]
    dst, dst_pts, M, Minv = transform_utils.image_alignment(src, src_pts, dsize=(400, 300), scale=(1.2, 1.2))
    src = image_utils.draw_image_contours(src, contours)
    src = image_utils.draw_landmark(src, [src_pts], color=(255, 0, 0), vis_id=True)
    dst = image_utils.draw_landmark(dst, [dst_pts], color=(0, 255, 0), vis_id=True)
    image_utils.cv_show_image("src", src, delay=10)
    image_utils.cv_show_image("dst", dst, delay=0)


if __name__ == '__main__':
    image_file = "../data/mask/mask5.jpg"
    test_transform(image_file)
