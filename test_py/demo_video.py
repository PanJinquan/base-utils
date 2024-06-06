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
    files = file_utils.get_files_lists(root, postfix=["*.avi", "*.mp4", "*.flv"])
    for video_file in files:
        print(video_file)
        video_utils.video2frames_similarity(video_file, out_dir=out, func=None, interval=20, thresh=0.3, vis=True)
        # video_utils.video2frames(video_file, out_dir=out, func=None, interval=20, vis=True)


if __name__ == "__main__":
    root = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/未归档（新增的未处理文件放这里）/20240508_广州16楼展厅2场数据/anwu_20240508indoor02/anwu_20240508indoor02_right.mp4"
    # out = root + "-frame"
    # out = os.path.join(os.path.dirname(root), "frame")
    out = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-cls2/手与表箱门有接触/手与表箱门有接触-v2/暗物16楼"
    video2frames_demo(root, out)
