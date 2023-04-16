# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils


def image_dir_mask(image_dir, info_file):
    names = file_utils.read_json_data(info_file)
    sub_list = file_utils.get_sub_paths(image_dir)
    others = []
    class_dict = names.copy()
    for sub in sub_list:
        old = sub.strip().lower()
        if old in names:
            new = names[old]["name"]
            old = os.path.join(image_dir, old)
            new = os.path.join(image_dir, new)
            if os.path.exists(old):
                os.rename(old, new)
        else:
            # class_dict[old] = {"name": ""}
            others.append(old)
    print(others)



if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/tmp/Medicine/dataset/train"
    info_file = "/home/dm/nasdata/dataset/tmp/Medicine/dataset/file.json"
    image_dir_mask(image_dir, info_file)
