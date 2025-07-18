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
    print(round(3.14159, 1))  # 输出: 3.1
    print(round(3.14159, 2))  # 输出: 3.14
    print(round(3.14159, 3))  # 输出: 3.142