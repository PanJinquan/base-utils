# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-07-10 17:47:57
    @Brief  :
"""
import os
import sys
import numpy as np
import cv2
from pybaseutils import image_utils, file_utils


def fatigue_driving(data_file, vis=False):
    data = np.load(data_file)
    for i in range(len(data)):
        image = data[i, :]
        image = np.array(image, dtype=np.float32) + 128
        image = np.array(image, dtype=np.uint8)
        if vis:
            image_utils.cv_show_image("image", image)
    return data


if __name__ == "__main__":
    data_file = "/home/PKing/nasdata/dataset/tmp/challenge/自动驾驶疲劳检测挑战赛/fatigue-driving/train.npy"
    fatigue_driving(data_file, vis=True)
