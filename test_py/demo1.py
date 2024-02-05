# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import re
import random
import types
import numpy as np
import time
import datetime
import threading
import torch
from pycocotools import cocoeval
from pybaseutils import image_utils, file_utils
from pybaseutils.singleton_utils import Singleton
import skimage.metrics as metrics

if __name__ == '__main__':
    file = "../data/mask.jpg"
    src = cv2.imread(file)
    for i in range(0, 1000000, 2):
        image = src.copy()
        image = image_utils.image_rotation(image, angle=i % 360, borderValue=(255, 255, 255))
        mask = image_utils.get_image_mask(image, inv=True)
        contours = image_utils.find_mask_contours(mask)
        points = image_utils.find_minAreaRect(contours)
        image = image_utils.draw_image_contours(image, contours=contours)
        image = image_utils.draw_key_point_in_image(image, key_points=points, vis_id=True)
        # image = image_utils.draw_image_contours(image, contours=points)
        image_utils.cv_show_image("mask", image)
