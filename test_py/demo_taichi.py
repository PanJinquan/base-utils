# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-09-08 09:22:01
    @Brief  :
"""

import time
import taichi as ti
import cv2
import numpy as np
from pybaseutils import time_utils, image_utils

ti.init(arch=ti.cpu)  # 添加了这行


# ti.init(arch=ti.gpu)  # 添加了这行


@ti.kernel
def image_demo(src: ti.types.ndarray(element_dim=1), dst: ti.types.ndarray(element_dim=1)):
    dst[:] = src[:]
    for i, j in ti.ndrange(h, w):
        dst[i, j] = src[i, j]


if __name__ == "__main__":
    image_file = "test.png"
    src = cv2.imread(image_file)
    h, w, c = src.shape
    dst = np.zeros(shape=src.shape, dtype=src.dtype)
    image_demo(src, dst)
    image_utils.cv_show_image("image", dst)
