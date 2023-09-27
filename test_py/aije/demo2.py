# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import random
import numpy as np
from pybaseutils import image_utils, file_utils
import cv2

if __name__ == '__main__':
    file_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v1/trainval.txt"
    file_utils.get_train_test_files(file_dir=file_dir, ratio=500)
