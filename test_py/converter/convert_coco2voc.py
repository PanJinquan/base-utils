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
from pybaseutils.dataloader import parser_coco_det
from pybaseutils.converter import build_voc, convert_coco2voc
from pybaseutils import file_utils, image_utils

if __name__ == "__main__":
    filename = "/home/PKing/nasdata/dataset/tmp/pen/dataset-pentip/test/coco_kps.json"
    out_xml_dir = os.path.join(os.path.dirname(filename), "VOC/Annotations")
    # out_image_dir = os.path.join(os.path.dirname(filename), "VOC/JPEGImages")
    out_image_dir = ""
    class_name = None
    convert_coco2voc.convert_coco2voc(filename, out_xml_dir, out_image_dir=out_image_dir,
                                      class_name=class_name,  rename="", vis=False)
