# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-05-23 11:24:37
    @Brief  :
"""
import os

import cv2
import numpy as np
from sympy.printing.pretty.pretty_symbology import center
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, numpy_utils
from pybaseutils.cvutils import corner_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    boxes = [[100, 100, 500, 600]]
    image_file = "/home/PKing/Pictures/DMovie/image-2025-05-12-13h52m05s385.jpg"
    image = cv2.imread(image_file)
    image = image_utils.draw_image_boxes(image, boxes, color=(255, 0, 0),thickness=6)
    # boxes = image_utils.extend_xyxy(boxes, scale=[0.9,0.9])
    boxes = image_utils.extend_xyxy(boxes, scale=[1.0, 0.9, 1.0, 1.0])
    image = image_utils.draw_image_boxes(image, boxes, color=(0, 255, 0),thickness=2)
    image = image_utils.cv_show_image("image", image)
