# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import cv2
from pybaseutils import image_utils, file_utils, json_utils

if __name__ == "__main__":
    file1 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval-image/1227.txt"
    file2 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval-image/crops-square"
    # file2 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval/train"
    class_names1 = file_utils.read_data(file1, split=None)[0]
    class_names1 = list(eval(class_names1))
    class_names2 = file_utils.get_sub_paths(file2)
    diff12 = list(set(class_names1) - set(class_names2))
    print("class_names1:{}".format(len(class_names1)))
    print("class_names2:{}".format(len(class_names2)))
    print(len(diff12), diff12)
