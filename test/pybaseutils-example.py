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
    # boxes2 = coords_utils.get_square_bboxes(boxes1,use_max=False)
    boxes2 = coords_utils.extend_xyxy(boxes1, scale=[2.0, 1.0])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_utils.draw_image_boxes(image, boxes1, thickness=4, color=(0, 255, 0))
    image = image_utils.draw_image_boxes(image, boxes2, thickness=2)
    image_utils.cv_show_image("image", image)
