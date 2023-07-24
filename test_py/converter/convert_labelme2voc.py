# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: AlphaPose
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-02-14 09:15:52
# --------------------------------------------------------
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.maker import maker_voc
from pybaseutils.maker.convert_labelme2voc import LabelMeDemo
from pybaseutils.dataloader import parser_labelme

if __name__ == "__main__":
    json_dir = "/home/PKing/nasdata/dataset/tmp/Drowsy-Driving/video/video-frame/json"
    out_root = os.path.dirname(json_dir)
    image_dir = os.path.join(out_root, "JPEGImages")
    class_dict = None
    # class_dict = {"person": "up", "squat": "bending", "fall": "down"}
    lm = LabelMeDemo(json_dir, image_dir)
    lm.convert_dataset2voc(out_root, class_dict=class_dict, vis=False, crop=True)
