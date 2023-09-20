# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import numpy as np
from pybaseutils import image_utils
import cv2

# 如算法返回angle=5,则需要对原图进行反向旋转对应的角度，可如下调用：
# image_rotation(image, angle=-5)
image = np.zeros(shape=(192, 160), dtype=np.uint8)

image_utils.cv_show_image("image", image)
