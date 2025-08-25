# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import os
import cv2
import random
import types
import torch
from typing import List, Tuple, Dict
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils, json_utils
from pybaseutils.cvutils import video_utils
import cv2
import re
import torch
import subprocess


def load_annotation(annot_file, image):
    data_info = json_utils.load_json(annot_file)
    annot = data_info['result']
    points, labels = [], []
    h, w = image.shape[:2]
    for res in annot:
        s = res['value']['width']
        w, h = res['original_width'], res['original_height']
        p = (res['value']['x'] / s, res['value']['y'] / s)
        n = res['value']['keypointlabels'][0]
        points.append(p)
        labels.append(n)
    # image = cv2.resize(image, (int(w), int(h)))
    return points, labels, image


def image_data(image_dir, annot_dir, vis=True):
    annot_list = file_utils.get_files_lists(annot_dir, postfix=None)
    for annot_file in annot_list:
        annot_file = "/home/PKing/nasdata/dataset/指针表计/labels_label_studio/823"
        print(annot_file)
        image_file = os.path.join(image_dir, f"{os.path.basename(annot_file)}.jpg")
        image = cv2.imread(image_file)
        points, labels, image = load_annotation(annot_file=annot_file, image=image)
        image = image_utils.draw_points_texts(image, points, labels, thickness=1, fontScale=1)
        if vis:
            image_utils.show_image("image", image, delay=0)


class Result(object):
    def __init__(self, boxes=[], label=[], score=[]):
        self.boxes = boxes
        self.score = score
        self.label = label

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v) if hasattr(self, k) else None

    def dict(self, keys: List=[]):
        out = self.__dict__
        if keys: out = {k: out.get(k, None) for k in keys}
        return out


if __name__ == '__main__':
    boxes = [(0, 0, 1, 1)]
    label = [1]
    score = [0.9]
    r = Result()
    print(r.dict())
    r.update(boxes=boxes, label=label, score=score)
    print(r.dict(keys=['boxes', "ffff"]))
