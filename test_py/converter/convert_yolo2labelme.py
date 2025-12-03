# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils import file_utils,yaml_utils
from pybaseutils.converter import build_labelme, convert_yolo2labelme

if __name__ == "__main__":
    filename = "/home/PKing/nasdata/release/edu-engineering/Pytorch-YOLO-Trainer/cfg/datasets/coco128-seg-local.yaml"
    config = yaml_utils.load_config(filename)
    class_name = list(config["names"].values())
    data_root="/home/PKing/nasdata/release/edu-engineering/datasets/coco128-labelme"
    convert_yolo2labelme.convert_yolo2labelme(data_root, class_name=class_name, prefix="train", task="seg",
                                              check=True,vis=False)
