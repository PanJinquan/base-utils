# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-04-12 15:58:07
    @Brief  :
"""
import os
import json
import cv2
import tt100k_utils
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.converter import build_voc


def parser_dataset(image_dir, anno_file, out_voc, vis=True):
    anno_info = file_utils.read_json_data(anno_file)
    for data in tqdm(anno_info):
        class_id = data['disease_class']
        image_name = data['image_ids']
        image_file = os.path.join(image_dir, image_name)
        if not os.path.exists(image_file): continue
        if vis:
            image = cv2.imread(image_file)
            image_utils.cv_show_image("image", image)


if __name__ == '__main__':
    out_voc = "/home/dm/nasdata/dataset/tmp/insects/-class/"
    anno_file = "/home/dm/nasdata/dataset/tmp/insects/ai_challenger_pdr2018_trainingset_20181023/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json"
    image_dir = "/home/dm/nasdata/dataset/tmp/insects/ai_challenger_pdr2018_trainingset_20181023/AgriculturalDisease_trainingset/images"
    parser_dataset(image_dir, anno_file, out_voc, vis=True)
