# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-25 17:42:55
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils.cvutils import video_utils
from pybaseutils import image_utils, file_utils


def video_converter(inp_dir, out_dir):
    files = file_utils.get_files_list(inp_dir, postfix=["*.dav"])
    for video_file in tqdm(files):
        save_video = file_utils.create_dir(out_dir, None, os.path.basename(video_file))
        save_video = save_video.replace(".dav", ".mp4")
        video_utils.video2video(video_file, save_video, interval=1, vis=False)


if __name__ == '__main__':
    inp_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/江门四维数据/江门四维2024-08-06/NVR录像文件"
    out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/技能人才系统_数据集管理/江门四维数据/江门四维2024-08-06/MP4录像文件"
    video_converter(inp_dir, out_dir)
