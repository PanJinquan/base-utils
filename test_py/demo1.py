# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc
from pybaseutils.dataloader import parser_labelme


def annotations_image(image_dir):
    file_list = file_utils.get_files_lists(image_dir)
    for image_file in tqdm(file_list):
        image = cv2.imread(image_file)
        cv2.imwrite(image_file, image)


if __name__ == '__main__':
    image_dir = "/home/PKingdddd/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det2/images"
    annotations_image(image_dir)