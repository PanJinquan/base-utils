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
import random
import json
from typing import Dict
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, base64_utils, time_utils
from pybaseutils.cvutils import video_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils.dataloader import parser_labelme
import xmltodict


class MultiViewer(object):
    def __init__(self, frame_info: Dict, key):
        self.key = key
        self.frame_info = frame_info

    def __enter__(self):
        return self.frame_info.get(self.key, [])

    def __exit__(self, type, value, trace):
        print("exit", type, value, trace)
        return "exit"


class Promote():
    def __init__(self, promote):
        self.promote = file_utils.read_data(promote, split=None)
        print("promote:{}".format(self.promote))

    def __len__(self):
        return len(self.promote)

    def __getitem__(self, index):
        data: str = self.promote[random.randint(0, len(self.promote) - 1)]
        data = data.format("ddd")
        return data


if __name__ == '__main__':
    image_file = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/第3批数据/aije-action-train-v22/江门_检查工器具_20240910_172411_6606_0000.jpg"
    boxe1 = [[600, 300, 850, 800]]
    image = image_utils.read_image(image_file)
    boxe2 = image_utils.get_square_boxes(boxe1, use_max=True, use_mean=True)
    image = image_utils.draw_image_boxes(image, boxe1, color=(255, 0, 0))
    image = image_utils.draw_image_boxes(image, boxe2, color=(0, 255, 0))
    image_utils.cv_show_image("image", image)
