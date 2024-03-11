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
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc

if __name__ == '__main__':
    data_root = "/home/PKing/nasdata/dataset/tmp/水表数字识别/水表数据集/Water-Meter-Det1/val"
    out_root = data_root
    file_utils.get_pair_files(data_root,
                              out_root=out_root,
                              image_sub="images",
                              label_sub="labels",
                              label_postfix="txt")
