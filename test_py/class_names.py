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


def get_class_diff_set():
    file1 = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/trainval/class_name3594_pun_openset.txt"
    file2 = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/trainval/similar/word.txt"
    words1 = file_utils.read_data(file1, split=None)
    words2 = file_utils.read_data(file2, split=None)
    diff = set(words2) - set(words1)
    print("diff：{}".format(list(diff)))


if __name__ == "__main__":
    get_class_diff_set()
