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
from typing import List, Tuple, Dict
import numpy as np
from typing import Callable
from PIL import Image
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils, json_utils, base64_utils
from pybaseutils.cvutils import video_utils
import cv2
import re
import torch
import subprocess


def mean_squared_error(x, y):
    l1 = np.sum(np.square(x - y))
    return l1 / x.size


if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.random((8, 1, 224, 224))
    y = np.random.random((8, 224, 224))
    out1 = mean_squared_error(x, y)
    print(out1)
    y = y[:, None, :]
    out2 = mean_squared_error(x, y)
    print(out2)
    image_utils.get_contours_iou()
