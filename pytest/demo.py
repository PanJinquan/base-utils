# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""

from pybaseutils import font_utils, file_utils, video_utils

if __name__ == "__main__":
    video_file = "/home/dm/nasdata/Project/3D/Camera-Calibration-Reconstruct-Cpp/docs/双目测距Demo.mp4"
    dst_file = "/home/dm/nasdata/Project/3D/Camera-Calibration-Reconstruct-Cpp/docs/双目测距Demo1.mp4"
    video_utils.video2video(video_file, dst_file,delay=5)
