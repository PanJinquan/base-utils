# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-08-24 11:05:52
    @Brief  :
"""
import os
import numpy as np
import itertools
from pybaseutils import file_utils, image_utils

if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/train-v2/dataset/all_action"
    pairs_file = os.path.join(image_dir, "pairs.txt")
    # pairs = file_utils.get_pair_data(image_dir, pair_num=-1)
    pairs = file_utils.get_pair_data(image_dir, pair_num=10000)
    file_utils.write_data(pairs_file, pairs)
