# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
from pybaseutils.converter.convert_labelme2voc import Labelme2VOC

if __name__ == "__main__":
    json_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/dataset-v4/json"
    out_root = os.path.dirname(json_dir)
    image_dir = os.path.join(out_root, "JPEGImages")
    class_dict = None
    # class_dict = {"person": "up", "squat": "bending", "fall": "down"}
    lm = Labelme2VOC(image_dir, json_dir)
    lm.build_dataset(out_root, class_dict=class_dict, vis=False, crop=False)
