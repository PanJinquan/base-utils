# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils
from pybaseutils.cvutils import video_utils


def video2frames_demo(root, out):
    files = file_utils.get_files_list(root, postfix=["*.avi", "*.mp4", "*.flv"])
    for video_file in files:
        print(video_file)
        # video_utils.video2frames_similarity(video_file, out_dir=out, func=None, interval=20, thresh=0.2,vis=True)
        video_utils.video2frames(video_file, out_dir=out, func=None, interval=15, vis=True)


if __name__ == "__main__":
    root = "/home/PKing/nasdata/dataset/tmp/Drowsy-Driving/video/videos"
    out = root + "-frame"
    video2frames_demo(root, out)
