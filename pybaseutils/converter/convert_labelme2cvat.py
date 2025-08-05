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
from pybaseutils.dataloader import parser_voc, parser_labelme
from pybaseutils.converter import build_voc, build_labelme, build_cvat
from pybaseutils import file_utils, image_utils, json_utils


def convert_labelme2cvat(anno_dir, image_dir="", vis=False):
    """
    将labelme文件转换为CVAT标注格式(LabelMe 3.0)
    :param anno_dir: 标注文件目录，
    :param image_dir: 图片文件目录，输出xml文件与anno_dir同目录
    :param vis:
    :return:
    """
    image_list = file_utils.get_images_list(image_dir)
    for image_file in tqdm(image_list):
        image_name = os.path.basename(image_file)
        json_file = os.path.join(anno_dir, image_name.split(".")[0] + ".json")
        xml_file = os.path.join(image_dir, image_name.split(".")[0] + ".xml")
        info = parser_labelme.parser_labelme(json_file, class_dict={})
        boxes, labels, points = info["boxes"], info["points"], info["labels"]
        image = cv2.imread(image_file)
        h, w = image.shape[:2]
        build_cvat.maker_cvat(xml_file, points, labels, image_name, image_size=[w, h])


if __name__ == "__main__":
    image_dir = "/home/PKing/Downloads/dataset-label/image"
    anno_dir = "/home/PKing/Downloads/dataset-label/json"
    convert_labelme2cvat(image_dir=image_dir, anno_dir=anno_dir, vis=False)
