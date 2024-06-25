# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, base64_utils, time_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils.dataloader import parser_labelme


@time_utils.performance()
def test_time1(image_file):
    size = file_utils.get_file_size(image_file)
    if size > 256:
        image = cv2.imread(image_file, cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_file, flags=cv2.IMREAD_COLOR)
    print("image1 size:{}".format(image.shape))
    # image_utils.cv_show_image("image", image)
    return image


@time_utils.performance()
def test_time2(image_file):
    image = cv2.imread(image_file)
    print("image2 size:{}".format(image.shape))
    # image_utils.cv_show_image("image", image)
    return image


if __name__ == '__main__':
    size = file_utils.get_file_size("/home/PKing/Downloads/images/test1.jpg")
    image_dir = "/home/PKing/Downloads/images"
    image_list = file_utils.get_files_lists(image_dir)
    image_list1 = image_list * 100
    image_list2 = image_list * 100
    random.shuffle(image_list1)
    random.shuffle(image_list2)
    for file1, file2 in tqdm(zip(image_list1, image_list2)):
        with time_utils.Performance("test_time2") as t:
            test_time2(file2)
        with time_utils.Performance("test_time1") as t:
            test_time1(file1)
