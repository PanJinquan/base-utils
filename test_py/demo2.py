# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import random
import types
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import video_utils
import cv2

if __name__ == '__main__':
    from pybaseutils.converter import convert_voc2yolo

    # 定义类别数
    class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                  "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                  "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # VOC数据目录
    data_root = "/path/to/VOC2007"
    # 保存输出yolo格式数据目录
    out_text_dir = os.path.join(data_root, "labels")
    # 开始转换,vis=True进行可视化
    convert_voc2yolo.convert_voc2yolo(data_root, out_text_dir, class_name=class_name, vis=True)