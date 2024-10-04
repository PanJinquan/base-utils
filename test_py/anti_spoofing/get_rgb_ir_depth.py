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
from tqdm import tqdm
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re


def check_file_exit(file_path):
    r = True
    for path in file_path:
        if not os.path.exists(path):
            print("not exist:{}".format(path))
            r = False
            break
    return r


def get_file_list_dataset(data_dir, image_name="color", other_name=["ir", "depth"], check=True):
    """
    :param data_dir:
    :param image_name:
    :param other_name:
    :param check:
    :return:
    """
    print(data_dir)
    image_list = file_utils.get_files_lists(data_dir, sub=True, postfix=file_utils.IMG_POSTFIX)
    image_list = [file for file in image_list if os.path.dirname(file).split(os.sep)[-1] == image_name]
    phase = os.path.basename(data_dir)
    content = []
    for image_file in tqdm(image_list):
        label = image_file.split(os.sep)[0]
        other_list = [image_file.replace(f"{image_name}/", f"{s}/") for s in other_name]
        pairs_list = [image_file] + other_list
        pairs_path = [os.path.join(data_dir, file) for file in pairs_list]
        if check and not check_file_exit(pairs_path):
            continue
        data = pairs_list + [label]
        content.append(data)
    content = sorted(content)
    print(f"have {len(content)} files")
    filename = os.path.join(os.path.dirname(data_dir), f"{phase}-new.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    data_dir = "/home/PKing/nasdata/FaceDataset/anti-spoofing/CASIA-SURF-CROP/test"
    # data_dir = "/home/PKing/nasdata/FaceDataset/anti-spoofing/CASIA-SURF-CROP/val"
    get_file_list_dataset(data_dir)
