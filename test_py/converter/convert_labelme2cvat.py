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
from pybaseutils.dataloader import parser_voc, parser_labelme
from pybaseutils.converter import convert_labelme2cvat
from pybaseutils import file_utils, image_utils, json_utils


if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-test/JPEGImages"
    anno_dir = os.path.join(os.path.dirname(image_dir), "json")
    print(image_dir)
    convert_labelme2cvat.convert_labelme2cvat(image_dir=image_dir,
                                              anno_dir=anno_dir,
                                              vis=True)
