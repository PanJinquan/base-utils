# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils, plot_utils, numpy_utils

if __name__ == '__main__':
    image_file = "/home/PKing/Downloads/20250213-085844.png"
    format = '%Y-%m-%d %H:%M:%S'  # 海康摄像头默认格式”
    fg_rgb = (0, 0, 0)
    bg_rgb = (125, 125, 125)
    image = cv2.imread(image_file, flags=cv2.IMREAD_UNCHANGED)
    image_utils.cv_show_image("image", image)
