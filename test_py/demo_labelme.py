# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 
    @Date   : 2023-06-29 18:19:18
    @Brief  :
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.converter import build_voc, convert_labelme2voc
from pybaseutils.dataloader import parser_labelme

if __name__ == "__main__":
    json_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-det/aije-action-train-v02-add/json"
    out_root = os.path.dirname(json_dir)
    image_dir = os.path.join(out_root, "JPEGImages")
    class_dict = None
    # class_dict = {"person": "up", "squat": "bending", "fall": "down"}
    lm = convert_labelme2voc.Labelme2VOC(json_dir, image_dir)
    lm.build_dataset(out_root, class_dict=class_dict, vis=False, crop=False)
