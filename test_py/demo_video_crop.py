# -*- coding: utf-8 -*-
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


def task(frame, **kwargs):
    frame = image_utils.resize_image(frame, size=(1280, None))
    return frame


if __name__ == "__main__":
    """视频截取片段"""
    video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/01-江门四维数据/01-37道考题视频/01-按考题分类/jiangmen_20241122Exam23Num03/平视.mp4"
    save_file = os.path.join(os.path.dirname(video_file), "crop.mp4")
    video_utils.video2video(video_file, save_file, start=60 * 11 + 4, end=60 * 11 + 35, task=task,interval=2)
