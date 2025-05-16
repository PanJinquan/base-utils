# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-04-15 19:51:59
# @Brief  :
# --------------------------------------------------------
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import pandas as pd
from pybaseutils.cvutils import video_utils
from tqdm import tqdm
from test_py.video_via import parse_via
from pybaseutils import json_utils, pandas_utils, image_utils, file_utils

if __name__ == '__main__':
    video_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/dataset-v1/video"
    annot_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/driving"
    output = video_dir + "-frame"
    via = parse_via.VIADataset(video_dir)
    for i in tqdm(range(len(via))):
        image_ids = via.image_ids[i]
        if "concat" in image_ids: continue
        annot_file = file_utils.change_postfix(image_ids.split("_")[-1], ".csv")
        annot_file = os.path.join(annot_dir, annot_file)
        video_file = os.path.join(video_dir, image_ids)
        via.video_extracter(video_file, annot_file, output=output, vis=False, delay=10)
