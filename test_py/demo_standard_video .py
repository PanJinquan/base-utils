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


def video_converter(root, out):
    files = file_utils.get_files_list(root, postfix=["*.avi"])
    for video_file in files:
        save_video = file_utils.create_dir(out, None, os.path.basename(video_file))
        save_video = save_video.replace(".avi", ".mp4")
        video_utils.video2video(video_file, save_video, interval=1, vis=False)


if __name__ == "__main__":
    root = "/media/dm/新加卷/SDK/project/Camera-Calibration-Reconstruct-Cpp/data/lenacv-video"
    out = "/media/dm/新加卷/SDK/project/Camera-Calibration-Reconstruct-Cpp/data/lenacv-video1"
    video_converter(root, out)
