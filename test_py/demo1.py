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


def demo():
    key = "multiview"
    frame_info = {"multiview": ["ABCD"]}
    with MultiViewer(frame_info=frame_info, key=key) as v:
        pass
    print("data")
    return "Z"


if __name__ == '__main__':
    print(demo())
