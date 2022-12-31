# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import time
import xmltodict
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def image_dir_move_file(image_dir, out_dir, shuffle=True):
    nums = 5000
    image_list = file_utils.get_images_list(image_dir)
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
        random.shuffle(image_list)
    image_list = image_list[:nums]
    for image_file in tqdm(image_list):
        # file_utils.copy_file_to_dir(image_file, out_dir)
        file_utils.move_file_to_dir(image_file, out_dir)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/plate/dataset/CCPD2019-voc/ccpd_base/plates"
    out_dir = "/home/dm/nasdata/dataset/csdn/plate/dataset/CCPD2019-voc/ccpd_base/plates-test"
    image_dir_move_file(image_dir, out_dir, shuffle=True)
