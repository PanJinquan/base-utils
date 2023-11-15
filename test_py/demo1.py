# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import re
import random
import types
import numpy as np
import time
import datetime
from pycocotools import cocoeval
from pybaseutils import image_utils, file_utils
import cv2


def custom_sort(*args):
    return True


oups = [
    {"image_id": 10.0},
    {"image_id": 10.5},
    {"image_id": 10.2},
    {"image_id": 10.3},
]

oups = sorted(oups, key=lambda x: x["image_id"], reverse=False)

print(oups)
