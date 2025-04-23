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
    dir = '/media/PKing/新加卷/SDK/base-utils/test_py/demo2.py'
    dir = '/media/PKing/新加卷/SDK/base-utils/test_py1/demo2.py'
    print(os.path.isdir(dir))
    print(os.path.isfile(dir))
