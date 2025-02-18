# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-01-31 15:45:48
    @Brief  :
"""
import os
import cv2
import random
import numpy as np
import time
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils
from pybaseutils.cvutils import video_utils


def video2frames(video_file, out_dir=None, func=None, interval=1, vis=True, delay=10):
    """
    视频抽帧图像
    :param video_file: 视频文件
    :param out_dir: 保存抽帧图像的目录
    :param func: 回调函数，对每一帧图像进行处理
    :param interval: 保存间隔
    :param vis: 是否可视化显示
    :return:
    """
    # name = os.path.basename(video_file).split(".")[0]
    name = file_utils.get_time(format="S")
    if not out_dir:  out_dir = os.path.join(os.path.dirname(video_file), name)
    video_cap = video_utils.get_video_capture(video_file)
    width, height, num_frames, fps = video_utils.get_video_info(video_cap)
    # if not interval: interval = fps
    interval = int(fps * 1.5)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    while True:
        count += 1
        isSuccess, frame = video_cap.read()
        if count % interval == 0:
            # 设置抽帧的位置
            # video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            if not isSuccess or 0 < num_frames < count: break
            if func:
                frame = func(frame)
            if vis:
                image_utils.cv_show_image("frame", frame, use_rgb=False, delay=delay)
            frame_file = os.path.join(out_dir, "{}_{:0=4d}.jpg".format(name, count))
            cv2.imwrite(frame_file, frame)
    video_cap.release()
    cv2.destroyAllWindows()


def video_capture(data_root, out_dir):
    video_list = file_utils.get_files_lists(data_root, postfix=file_utils.VIDEO_POSTFIX,
                                            subname="", shuffle=False, sub=False)
    for video_file in tqdm(video_list):
        if 'monitor' in video_file:
            continue
        elif video_file.endswith("_color.avi"):
            out = os.path.join(out_dir, "color")
            video2frames(video_file, out_dir=out, func=None, interval=50, vis=True, delay=10)
        elif video_file.endswith("_ir.avi"):
            out = os.path.join(out_dir, "ir")
            video2frames(video_file, out_dir=out, func=None, interval=50, vis=True, delay=10)


if __name__ == '__main__':
    data_root = "/home/PKing/nasdata/FaceDataset/anti-spoofing/DMAI_FASD/orig/train/fake_part"
    out_dir = "/home/PKing/nasdata/FaceDataset/anti-spoofing/DMAI_FASD/orig/train-rgb-ir/fake_part"
    video_capture(data_root, out_dir=out_dir)
