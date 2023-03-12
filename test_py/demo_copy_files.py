# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import os
import time
import xmltodict
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def image_dir_move_file(image_dir, out_dir, max_nums=None, shuffle=True):
    image_list = file_utils.get_images_list(image_dir)
    image_list = file_utils.get_sub_list(image_list, image_dir)
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
        random.shuffle(image_list)
    if max_nums:
        image_list = image_list[:min(max_nums, len(image_list))]

    for image_name in tqdm(image_list):
        src_file = os.path.join(image_dir, image_name)
        out_file = os.path.join(out_dir, image_name)
        # file_utils.copy_file(src_file, out_file)
        file_utils.move_file(src_file, out_file)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/emotion/emotion-domestic/train"
    out_dir = "/home/dm/nasdata/dataset/csdn/emotion/emotion-domestic/test"
    image_dir_move_file(image_dir, out_dir, max_nums=5000, shuffle=True)
