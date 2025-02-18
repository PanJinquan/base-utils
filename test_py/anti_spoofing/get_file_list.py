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
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re


def get_file_list_dataset(data_dir, subs=[]):
    files = file_utils.get_files_lists(data_dir, sub=True, postfix=file_utils.IMG_POSTFIX)
    # files = file_utils.get_files_lists(data_dir, sub=True, postfix=file_utils.VIDEO_POSTFIX)
    phase = os.path.basename(data_dir)
    content = []
    for path in files:
        label = path.split("/")[0]
        if subs and label not in subs: continue
        text = [os.path.join(phase, path), label]
        content.append(text)
    content = sorted(content)
    print(f"have {len(content)} files")
    filename = os.path.join(os.path.dirname(data_dir), f"{phase}-new.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    data_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/train"
    get_file_list_dataset(data_dir)
