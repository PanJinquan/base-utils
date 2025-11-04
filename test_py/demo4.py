# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
from pybaseutils.converter import convert_labelme2voc
from pybaseutils import time_utils


def video_generator(video: int | str):
    """
    :param video: 视频路径或摄像头索引
    :return:
    """
    if video is None:
        return None, None
    if len(video) == 1: video = int(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"无法打开视频/摄像头: {video}")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 默认30fps
    delay = 1.0 / fps
    try:
        while True:
            ret, src = cap.read()
            if not ret:  break
            src = src[:, :, ::-1]
            src, out = src,src
            yield src, out  # RGB image
    finally:
        cap.release()
    return None, None

if __name__ == "__main__":
    image_file = "/home/PKing/nasdata/Project/LLM/MLLM-Factory/data/test.mp4"
    data = video_generator(video=image_file)
    print(data)
    for src, out in data:
        print(type(src), type(out))

