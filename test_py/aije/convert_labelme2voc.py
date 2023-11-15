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
    json_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v2/json"
    data_root = os.path.dirname(json_dir)
    image_dir = os.path.join(data_root, "JPEGImages")
    class_dict = {"身穿工作服": "person", "未穿工作服": "person"}
    out_root = os.path.join(data_root, "person")
    out_image_dir = os.path.join(data_root, "person")
    lm = Labelme2VOC(image_dir, json_dir, class_name=class_dict,shuffle=False)
    lm.build_dataset(out_root, class_dict={}, out_img=True, vis=False, crop=False)
