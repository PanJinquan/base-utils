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
    """
    获得数据列表，格式
    file/path/image1 label-name1
    file/path/image2 label-name2
    :param image_dir:
    :param subs:
    :return:
    """
    # files = file_utils.get_files_lists(image_dir, sub=True, postfix=file_utils.IMG_POSTFIX)
    files = file_utils.get_files_lists(image_dir, sub=True, postfix=file_utils.AUDIO_POSTFIX)
    sub = os.path.basename(image_dir)
    content = []
    for path in files:
        label = path.split("/")[0]
        if subs and label not in subs: continue
        text = [os.path.join(sub, path), label]
        content.append(text)
    content = sorted(content)
    filename = os.path.join(os.path.dirname(image_dir), f"{sub}.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/tmp/tmp/challenge/旋转机械故障诊断挑战赛/旋转机械故障诊断挑战赛公开数据/test"
    get_file_list_dataset(image_dir)
