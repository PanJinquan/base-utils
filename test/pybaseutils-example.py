# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-04-25 16:20:17
    @Brief  :
"""
import copy
import cv2
import numpy as np
from pybaseutils import image_utils, file_utils, debug, coords_utils



if __name__ == "__main__":
    file = "../data/test_image/grid1.png"
    # file = "../data/test_image/grid2.png"
    # dsize = (400, 200)
    dsize = (200, 400)
    image = cv2.imread(file)
    boxes1 = [[100, 100, 200, 300], [400, 200, 450, 300]]
    boxes1 = np.asarray(boxes1)
    # for i in range(1,200):
    o = get_points_section(start=0, end=225, nums=20, dtype=np.int)
    print(len(o))
    print(o)