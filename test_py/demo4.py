# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
from pybaseutils.converter import convert_labelme2voc
from pybaseutils import time_utils, image_utils, file_utils, json_utils
fontScale=2.0
alpha = 0.
if __name__ == "__main__":
    image_file = "../data/test_image/grid1.png"

    for i in range(100):
        image = cv2.imread(image_file)
        boxes = [[0, i, 500, 300]]
        # texts = ["ABC"] * len(boxes)
        texts = ["ABC你是一名学生"] * len(boxes)
        print(boxes)
        image = image_utils.draw_image_boxes_texts(image, boxes, texts, thickness=1, fontScale=fontScale, alpha=alpha,drawType="simple")
        image_utils.show_image("ch", image)
        image = cv2.imread(image_file)
        image = image_utils.draw_image_boxes_texts(image, boxes, texts, thickness=1, fontScale=fontScale, alpha=alpha,drawType="en")
        image_utils.show_image("en", image)
