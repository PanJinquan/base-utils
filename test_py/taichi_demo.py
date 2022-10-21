# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-09-08 09:22:01
    @Brief  :
"""

import time
import taichi as ti
import cv2
from pybaseutils import time_utils

ti.init(arch=ti.cpu)  # 添加了这行


# ti.init(arch=ti.gpu)  # 添加了这行


@ti.kernel
def calc_pi(image:ti.Matrix, num: ti.f32) -> ti.f32:
    image_ = image + image
    sum = 0.0
    for i in range(num):
        n = 2 * i + 1
        sum += pow(-1.0, i) / n
    return sum * 4


if __name__ == "__main__":
    image_file = "test.png"
    image = cv2.imread(image_file)
    for i in range(20):
        with time_utils.Performance("PI"):
            calc_pi(image=image, num=100000)
