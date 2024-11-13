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
import torch
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re

if __name__ == '__main__':
    accs = np.asarray([0.6205, 0.5752, 0.9296, 0.9906, 0.4833, 0.0756, 0.000001, 0, 0, 0.8205, 0.9205])
    print(accs,np.mean(accs))
    accs = accs[accs > 0]
    print(accs,np.mean(accs))
