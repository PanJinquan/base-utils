# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import random
import types
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import video_utils
import cv2

if __name__ == '__main__':
    video_file = "/home/PKing/Downloads/kunkun_cut.mp4"
    saves_file = "/home/PKing/Downloads/kunkun_res.mp4"
    video_cap = video_utils.video_iterator(video_file, saves_file, start=4, end=10)
    for data_info in video_cap:
        frame = data_info["frame"]
        frame = cv2.resize(frame, dsize=(100, 100))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data_info["frame"] = frame
