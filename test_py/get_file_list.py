# -*-coding: utf-8 -*-
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
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re


def get_file_list_dataset(image_dir, subs=[]):
    files = file_utils.get_files_lists(image_dir, sub=True)
    sub = os.path.basename(image_dir)
    content = []
    for path in files:
        label = path.split("/")[0]
        if label not in subs: continue
        text = [os.path.join(sub, path), label]
        content.append(text)
    content = sorted(content)
    filename = os.path.join(os.path.dirname(image_dir), f"{sub}.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/test"
    get_file_list_dataset(image_dir)
