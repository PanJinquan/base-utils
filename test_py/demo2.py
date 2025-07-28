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
import subprocess

if __name__ == '__main__':
    import numpy as np
    from scipy.ndimage import median_filter

    # 原始数据（含噪声）
    data = [1, 2, 3, 50, 5, 6, 7, 800, 9, 10]

    # 中值滤波（窗口大小=3）
    smoothed_data = median_filter(data, size=3)

    print("原始数据:", data)
    print("中值滤波:", smoothed_data)