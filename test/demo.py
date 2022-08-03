# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import image_utils

image = [[0, 0, 255, 255]] * 4
image = np.asarray(image, dtype=np.uint8)
INTER_NEAREST = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_NEAREST)
INTER_LINEAR = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_LINEAR)
INTER_AREA = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_AREA)
INTER_CUBIC = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)
image_utils.cv_show_image("Origin(4×4)", image, delay=1)
image_utils.cv_show_image("INTER_NEAREST(8×8)", INTER_NEAREST, delay=1)
image_utils.cv_show_image("INTER_LINEAR(8×8)", INTER_LINEAR, delay=1)
image_utils.cv_show_image("INTER_AREA(8×8)", INTER_AREA, delay=1)
image_utils.cv_show_image("INTER_CUBIC(8×8)", INTER_CUBIC, delay=0)
