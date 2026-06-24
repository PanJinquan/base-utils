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
    video_file = "/home/PKing/Pictures/ailt-video.mp4"
    width, height, num_frames, fps = video_utils.get_video_info(video_file)
    clip_time = (13.5, 19.0)
    clip_index = (clip_time[0] * fps, clip_time[1] * fps)
    # video_utils.segment_video(video_file, clip_time=clip_time)
    video_utils.segment_video(video_file, clip_index=clip_index)
