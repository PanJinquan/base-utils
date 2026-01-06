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
import numbers
from typing import List, Tuple, Dict
import numpy as np
from typing import Callable
from PIL import Image
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils, json_utils, base64_utils
import numpy as np
import cv2
import math

import threading
import time

import numpy as np

from test_py.aije.示例代码.MobileNetV2 import output


def cal_contour_mean(image, contours):
    """
    计算指定轮廓内像素的平均值
    :param image:
    :param contours:
    :return:
    """
    output = []
    for c in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)  # 填充指定轮廓（白色）
        mean = cv2.mean(image, mask=mask)  # 计算掩膜内像素的平均值
        output.append(mean)
    return output


if __name__ == '__main__':
    image_file = "/home/PKing/nasdata/dataset-dmai/AILT/ailt-line/dataset-20251216-test/images/20251216_191638_主视_13300.jpg"
    image = cv2.imread(image_file)
    centers = [[100, 200], [300, 400], [500, 600]]
