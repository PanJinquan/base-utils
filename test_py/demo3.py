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

fontScale = 0.8
if __name__ == "__main__":
    image_file = "../data/test_image/grid1.png"
    image = cv2.imread(image_file)
    boxes = [[10, 20, 90, 100], [100, 100, 200, 150], [150, 200, 250, 250]]
    conts = image_utils.boxes2polygons(boxes)
    names = ["person", "cat", "person"]
    label = [0, 1, 0]
    class_name = ["person", "cat"]
    # image = image_utils.draw_image_boxes_labels_texts(image, boxes, label, texts=names,thickness=1,
    #                                             fontScale=fontScale, drawType="en")
    image = image_utils.draw_image_boxes_texts(image, boxes, names, thickness=1, fontScale=fontScale, drawType="en")
    # image = image_utils.draw_image_contours(image, conts, names, thickness=1)
    # image = image_utils.draw_image_contours_labels_texts(image, conts, label, names, thickness=1)
    image_utils.show_image("en", image)
