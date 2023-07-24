# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import os
import time
import xmltodict
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def image_dir_move_file(voc_root, out_dir, max_nums=500, move=True, shuffle=True):
    image_dir = os.path.join(voc_root, "JPEGImages")
    annos_dir = os.path.join(voc_root, "Annotations")
    json_dir = os.path.join(voc_root, "json")
    image_list = file_utils.get_files_lists(image_dir, postfix=file_utils.IMG_POSTFIX + ["*.JPG"])
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
        random.shuffle(image_list)
    image_list = image_list[:min(max_nums, len(image_list))]
    for img_file in tqdm(image_list):
        image_id = os.path.basename(img_file).split(".")[0]
        xml_file = os.path.join(annos_dir, "{}.xml".format(image_id))
        json_file = os.path.join(json_dir, "{}.json".format(image_id))
        if os.path.exists(img_file) and os.path.exists(xml_file):
            # file_utils.move_file_to_dir(image_file, out_dir)
            if move:
                file_utils.move_file_to_dir(img_file, file_utils.create_dir(out_dir, "JPEGImages"))
                file_utils.move_file_to_dir(xml_file, file_utils.create_dir(out_dir, "Annotations"))
            else:
                file_utils.copy_file_to_dir(img_file, file_utils.create_dir(out_dir, "JPEGImages"))
                file_utils.copy_file_to_dir(xml_file, file_utils.create_dir(out_dir, "Annotations"))
            if os.path.exists(json_file):
                if move:
                    file_utils.move_file_to_dir(json_file, file_utils.create_dir(out_dir, "json"))
                else:
                    file_utils.copy_file_to_dir(json_file, file_utils.create_dir(out_dir, "json"))


if __name__ == "__main__":
    voc_root = "/home/PKing/nasdata/dataset/tmp/drowsy-driving/drowsy-driving/Drowsy-Driving-Det1/trainval"
    out_dir = "/home/PKing/nasdata/dataset/tmp/drowsy-driving/drowsy-driving/Drowsy-Driving-Det1/test"
    image_dir_move_file(voc_root, out_dir, max_nums=500, shuffle=True)
