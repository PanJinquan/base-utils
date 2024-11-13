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


def build_dataset(anno_dir, class_dict=None):
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v1-det/dataset-v01/json"
    data_root = os.path.dirname(anno_dir)
    image_dir = os.path.join(data_root, "JPEGImages")
    # class_dict = {"身穿工作服": "person", "未穿工作服": "person"}
    # out_root = os.path.join(data_root, "person")
    # out_image_dir = os.path.join(data_root, "person")
    print(anno_dir)
    lm = Labelme2VOC(image_dir, anno_dir, class_name=class_dict, shuffle=False)
    lm.build_dataset(data_root, class_dict={}, out_img=False, vis=False, crop=False)


def build_datasets():
    anno_dirs = [
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v21/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v22/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v23/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v24/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v25/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v26/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v27/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v28/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v29/json',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v29-test/json',
    ]
    for anno_dir in anno_dirs:
        build_dataset(anno_dir)


if __name__ == "__main__":
    # 将AIJE项目数据集，转换为VOC数据集
    build_datasets()
