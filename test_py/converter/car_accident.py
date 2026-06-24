# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-06-23 17:33:01
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
from pybaseutils import json_utils, pandas_utils, image_utils, file_utils
from pybaseutils.cvutils import video_utils

class_maps = {
    "before crush": {"color": (255, 0, 0)},
    "crush moment": {"color": (0, 255, 0)},
    "after crush": {"color": (0, 0, 255)},
}


def parser_video(video_file, csv_file):
    print("video_file", video_file)
    print(" csv_file", csv_file)
    pd_data = pandas_utils.read_csv(csv_file, sep=",")
    pd_dict = pandas_utils.df2dict(pd_data, orient="index")
    for info in pd_dict.values():
        index = info['frame_num']
        label = info['type'].strip()
        video_cap = video_utils.get_video_capture(video_file)
        # 设置抽帧的位置
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video_cap.read()
        if not ret: break
        color = class_maps[label]["color"]
        text = "{:3d}-{}".format(index, label)
        frame = image_utils.draw_text(frame, (0, 0), text, color=color, fontScale=1, thickness=2)
        image_utils.show_image("image", frame, delay=20)


def parser_video_dir(video_dir, csv_dir):
    video_files = file_utils.get_files_list(video_dir, postfix=file_utils.VIDEO_POSTFIX)
    for video_file in video_files:
        name, ext = os.path.splitext(os.path.basename(video_file))
        csv_file = os.path.join(csv_dir, name.replace("-", "_") + ".csv")
        if not os.path.exists(csv_file):
            print("not exists:{}".format(csv_file))
            continue
        parser_video(video_file, csv_file)


if __name__ == '__main__':
    video_dir = '/home/PKing/nasdata/tmp/tmp/car-accident/Crash dataset/video/'
    csv_dir = "/home/PKing/nasdata/tmp/tmp/car-accident/Crash dataset/label/csv"
    parser_video_dir(video_dir, csv_dir)
