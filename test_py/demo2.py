# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import os
import cv2
import random
import types
import torch
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils
from pybaseutils.cvutils import video_utils
import cv2
import re
import torch

if __name__ == '__main__':
    image_file = "/home/PKing/nasdata/release/eduea/eduea-calligraphy-preprocessing/test/image_bug.png"
    gray = cv2.imread(image_file,flags=cv2.IMREAD_UNCHANGED)
    edges = cv2.Canny(gray, threshold1=0, threshold2=255, apertureSize=3)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
