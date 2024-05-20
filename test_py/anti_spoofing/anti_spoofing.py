# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-01-31 15:45:48
    @Brief  :
"""
import sys
import os

sys.path.insert(0, "/home/PKing/nasdata/release/detector/object-detector")
from libs.detector.detector import Detector

import cv2
import random
import numpy as np
import time
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils


def parser_anti_spoofing(data_root, class_dict={}, conf_thresh=0.9, nms_thresh=0.3, vis=True):
    # detector = Detector(conf_thresh=conf_thresh, nms_thresh=nms_thresh, detect_type="mtcnn")
    detector = Detector(conf_thresh=conf_thresh, nms_thresh=nms_thresh, detect_type="dfsd")
    image_dir = data_root
    image_list = file_utils.get_files_lists(data_root, sub=True)
    content = []
    for path in image_list:
        print(path)
        try:
            name = path.split("/")[0]
            image_file = os.path.join(image_dir, path)
            image = cv2.imread(image_file)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets, _ = detector.detect(rgb, vis=False)
            if len(dets) > 0:
                boxes = dets[0:1, 0:4].tolist()
                labels = [class_dict[name]]
                text = [[path, l] + b for b, l in zip(boxes, labels)]
                content += text
            else:
                boxes,  labels = [], []
            if vis:
                image = image_utils.draw_image_bboxes_text(image, boxes, labels)
                image_utils.cv_show_image("image", image, delay=5)
        except Exception as e:
            print(e)
    filename = os.path.join(image_dir, "data_label.txt")
    file_utils.write_data(filename, content, split=",")


if __name__ == '__main__':
    class_dict = {"real_part": 0, "fake_part": 1}  # 0-live 1-spoof label
    data_root = "/home/PKing/nasdata/FaceDataset/anti-spoofing/DMAI_FASD/orig/train"
    # data_root = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    parser_anti_spoofing(data_root, class_dict=class_dict)
