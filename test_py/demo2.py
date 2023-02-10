# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils, font_utils, json_utils

if __name__ == "__main__":
    params = [
        {"type": 1},
        {"type": 2},
        {"type": 11},
        {"type": 22},
        {"type": 111},
        {"type": 3},
        {"type": 4},
        {"type": 33},
        {"type": 333},
        {"type": 2},
        {"type": 22},
        {"type": 222},
    ]
    group = {"A": [], "B": [], "C": [], "D": []}
    index = []
    for i, data in enumerate(params):
        if data["type"] < 10:
            index.append(["A", len(group["A"])])
            group["A"].append(data)
        elif 10 < data["type"] < 100:
            index.append(["B", len(group["B"])])
            group["B"].append(data)
        elif 100 < data["type"] < 1000:
            index.append(["C", len(group["C"])])
            group["C"].append(data)
    print("group:{}".format(group))
    print("index :{}".format(index))
    output = json_utils.get_values(group, keys=index)
    print("params:{}".format(params))
    print("output:{}".format(output))

