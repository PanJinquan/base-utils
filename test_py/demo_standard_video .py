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


def video_converter(root, out):
    files = file_utils.get_files_list(root, postfix=["*.avi", "*.dav"])
    for video_file in files:
        save_video = file_utils.create_dir(out, None, os.path.basename(video_file))
        save_video = save_video.replace(".avi", ".mp4").replace(".dav", ".mp4")
        video_utils.video2video(video_file, save_video, interval=1, vis=False)


def resize_video():
    video_file = "/media/PKing/新加卷/SDK/project/cv-sdk-ncnn/cv-sdk-ncnn/data/video/face_person.avi"
    save_video = "/media/PKing/新加卷/SDK/project/cv-sdk-ncnn/cv-sdk-ncnn/data/video/face_person.mp4"
    video_utils.resize_video(video_file, save_video, size=(None, 640), interval=1, vis=True, delay=10)


def rotate_video():
    video_file1 = "/home/PKing/Downloads/20240605-155224.mp4"
    video_file2 = "/home/PKing/Downloads/result.mp4"
    video_utils.video2video(video_file1, video_file2, task=video_utils.rotation_task, delay=10)


if __name__ == "__main__":
    # root = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/01-江门四维数据/01-37道考题视频/江门四维2024-12-05-冲击试验/2024-12-05-冲击试验/dav"
    # out = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/01-江门四维数据/01-37道考题视频/江门四维2024-12-05-冲击试验/2024-12-05-冲击试验/mp4"
    # video_converter(root, out)
    resize_video()
    # rotate_video()
