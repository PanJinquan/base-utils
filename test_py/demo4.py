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
from pybaseutils import time_utils, image_utils, file_utils

if __name__ == "__main__":
    image_dir = "/home/PKing/Downloads/image/摄像头ID.txt"
    lines = file_utils.read_file(image_dir)
    print(lines)
