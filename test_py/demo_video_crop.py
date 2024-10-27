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


def video2crop_demo(video_file, save_file):
    video_utils.video2video(video_file, save_file, start=4, end=16)

if __name__ == "__main__":
    video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/江门四维数据/【01】考试采集视频/江门四维2024-07-16/【01】已分类视频/10KV架空线路停电更换耐张绝缘子/俯视.mp4"
    save_file = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/江门四维数据/【01】考试采集视频/江门四维2024-07-16/【01】已分类视频/10KV架空线路停电更换耐张绝缘子/俯视-crop.mp4"
    video2crop_demo(video_file, save_file)
