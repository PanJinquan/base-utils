# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-01-31 15:45:48
    @Brief  :
"""
import os
import cv2
import random
import numpy as np
import time
import traceback
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils

class_dict = {0: "real", 1: "spoof"}


def read_annotation(image_file, boxes_file, annotation):
    image = cv2.imread(image_file)
    real_h, real_w = image.shape[:2]
    rects = file_utils.read_data(boxes_file, split=" ")
    rects = np.asarray(rects)
    rects = rects[0:1, 0:4] * (real_w, real_h, real_w, real_h) / 224
    boxes = image_utils.xywh2xyxy(rects)
    boxes = np.asarray(boxes, dtype=np.int32)
    boxes = boxes.tolist()
    labels = [annotation[43]]
    labels = [class_dict[l] for l in labels]
    return image, boxes, labels


def CelebA_Spoof_Dataset(anno_file, image_dir, vis=False):
    """
    :param anno_dir:
    :param image_dir:
    :return:
    """
    # value: label of image;
    #   [0:40]: face attribute labels,
    #   [40]: spoof type label,
    #   [41]: illumination label,
    #   [42]: Environment label
    #   [43]: 0-live 1-spoof label
    content = []
    anno_info: dict = json_utils.read_json_data(anno_file)
    path_list = list(anno_info.keys())
    for path in tqdm(path_list):
        try:
            anno = anno_info[path]
            postfix = os.path.basename(path).split(".")[-1]
            path = path.replace("Data/", "")
            image_file = os.path.join(image_dir, path)
            boxes_file = image_file.replace(f".{postfix}", f"_BB.txt")
            image, boxes, labels = read_annotation(image_file, boxes_file, annotation=anno)
            text = [[path, l] + b for b, l in zip(boxes, labels)]
            content += text
            # break
            if vis:
                image = image_utils.draw_image_bboxes_text(image, boxes, labels)
                image_utils.cv_show_image("image", image)
        except:
            print(path)
            traceback.print_exc()
        # break
    content = sorted(content)
    filename = os.path.join(image_dir, "label.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    # anno_file = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/CelebA-Spoof/CelebA_Spoof/metas/intra_test/train_label.json"
    anno_file = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/CelebA-Spoof/CelebA_Spoof/metas/intra_test/test_label.json"
    image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/CelebA-Spoof/CelebA_Spoof/Data"
    CelebA_Spoof_Dataset(anno_file, image_dir, vis=False)
