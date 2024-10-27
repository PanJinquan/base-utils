# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import random
import types
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re


def equalizeHist(image):
    image = cv2.equalizeHist(image)
    return image


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/暗光数据"
    image_list = file_utils.get_images_list(image_dir)
    for image_file in image_list:
        src = cv2.imread(image_file)
        dst = equalizeHist(src)
        image_utils.cv_show_image("dst", dst, delay=10)
        image_utils.cv_show_image("src", src, delay=0)
