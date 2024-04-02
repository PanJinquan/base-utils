# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import random
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
import cv2

if __name__ == '__main__':
    data_file = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v1/val.txt"
    out_root = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-test"
    file_map = {"Annotations": "xml", "json": "json", "JPEGImages": None}
    file_utils.copy_move_voc_dataset(data_file, out_root=out_root, file_map=file_map, move=True)
