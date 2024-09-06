# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_voc
from pybaseutils.converter import build_voc, build_labelme, convert_cvat2labelme
from pybaseutils import file_utils, image_utils, json_utils

if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/aije-action-train-v02/JPEGImages"
    anno_dir = os.path.join(os.path.dirname(image_dir), "xml")
    json_dir = os.path.join(os.path.dirname(image_dir), "json")
    convert_cvat2labelme.convert_cvat2labelme(image_dir=image_dir, anno_dir=anno_dir, out_dir=json_dir,
                                              thickness=1, fontScale=2.0, vis=True)
