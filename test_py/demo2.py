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

if __name__ == '__main__':
    file = "/home/PKing/nasdata/dataset-dmai/AIJE/方案图/pose2/result.mp4"
    video_utils.video2frames(file)

