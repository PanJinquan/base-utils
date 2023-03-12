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


def image_dir_mask(image_dir):
    images = file_utils.get_images_list(image_dir)
    item_list = file_utils.get_sub_list(images, image_dir)
    # item_list = [[path, os.path.dirname(path)] for path in images]
    filename = file_utils.create_dir(image_dir, None, "data.txt")
    file_utils.write_list_data(filename, item_list)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/emotion-dataset/MMA FACIAL EXPRESSION/MMAFEDB/valid"
    image_dir_mask(image_dir)
