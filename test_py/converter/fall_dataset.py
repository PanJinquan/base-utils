# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-12-12 14:23:50
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
from collections import defaultdict
from pybaseutils import file_utils, pandas_utils, json_utils, image_utils, geometry_tools
from pybaseutils.cvutils import video_utils


def clip_label_dataset(anno_file: str, data_dir: str, save_dir: str = None):
    df = pandas_utils.read_csv(anno_file, sep=",")
    data_info = pandas_utils.df2dict(df, orient="index")
    # 遍历每一行数据
    outs_data = []
    for i, data in data_info.items():
        name = "chute{:02d}/cam{}.avi".format(int(data['chute']), int(data['cam']))
        file = os.path.join(data_dir, name)
        if not os.path.exists(file):
            print("{} not exist".format(file))
            continue
        start = int(data['start'])
        end = int(data['end'])
        label = int(data['label'])
        print("file:{}, {}".format(file, data))
        video_name = name.replace("/", "_").split(".")[0] + "_{:0=4d}_{}.mp4".format(i, label)
        video_name = os.path.join("video", video_name)
        save_video = os.path.join(save_dir, video_name)
        video_utils.video_capture(file, save_video=save_video, vis=False, clip=(start, end), title="frame", delay=5)
        info = {"file": video_name, "label": label}
        outs_data.append(info)
    json_utils.save_json(os.path.join(save_dir, "label_dataset.json"), outs_data)
    return outs_data


def get_ori_data_label(anno_file: str, data_dir: str):
    df = pandas_utils.read_csv(anno_file, sep=",")
    data_info = pandas_utils.df2dict(df, orient="index")
    # 遍历每一行数据
    outs_data = defaultdict(list)
    for i, data in data_info.items():
        name = "chute{:02d}/cam{}.avi".format(int(data['chute']), int(data['cam']))
        file = os.path.join(data_dir, name)
        if not os.path.exists(file):
            print("{} not exist".format(file))
            continue
        start = int(data['start'])
        end = int(data['end'])
        label = int(data['label'])
        print("file:{}, {}".format(file, data))
        outs_data[name].append(data)
    json_utils.save_json(os.path.join(data_dir, "video_dataset.json"), outs_data)
    return outs_data


def clip_video_dataset(anno_file: str, data_dir: str, save_dir: str = None):
    df = pandas_utils.read_csv(anno_file, sep=",")
    data_info = pandas_utils.df2dict(df, orient="index")
    # 遍历每一行数据
    outs_data = []
    for i, data in data_info.items():
        name = "chute{:02d}/cam{}.avi".format(int(data['chute']), int(data['cam']))
        file = os.path.join(data_dir, name)
        if not os.path.exists(file):
            print("{} not exist".format(file))
            continue
        start = int(data['start'])
        end = int(data['end'])
        label = int(data['label'])
        print("file:{}, {}".format(file, data))
        video_name = name.replace("/", "_").split(".")[0] + "_{:0=4d}_{}.mp4".format(i, label)
        video_name = os.path.join("video", video_name)
        save_video = os.path.join(save_dir, video_name)


def clip_time_video(anno_file: str, save_dir: str = None, chute=""):
    video_info = json_utils.load_json(anno_file)
    if chute: video_info = {k: v for k, v in video_info.items() if k.startswith(chute)}
    data_dir = os.path.dirname(anno_file)
    outs_data = []
    for name, clip_list in video_info.items():
        file = os.path.join(data_dir, name)
        if not os.path.exists(file):
            print("{} not exist".format(file))
            continue
        w, h, num_frames, fps = video_utils.get_video_info(file)
        clip_list = sorted(clip_list, key=lambda x: int(x['start']))
        clip_nums = len(clip_list)
        offs = 0
        count = 0
        for i, info in enumerate(clip_list):
            print("file:{}, {}".format(file, info))
            label = int(info['label'])
            clip = (int(info['start']), int(info['end']))
            if i == 0 and label == 0:  # TODO 第一个clip
                clip = (clip[0] - fps * 1.5, clip[1])
            elif i == clip_nums - 1 and label == 0:  # TODO 最后一个clip
                if offs > 0: clip = (offs, clip[1])
                clip = (clip[0], clip[0] + fps * 2)
            if 0 < i < clip_nums and label == 1:  # TODO 倒下label=1，躺着label=2
                clip = (clip[0] - int(0.1 * fps), clip[1] + int(1.5 * fps))
                offs = clip[1] + int(1.5 * fps)
            t = (clip[1] - clip[0]) / fps * 4  # 原始视频fps是120，4倍保存
            clips = [clip]
            if t > 4:
                n = t // 3
                cuts = geometry_tools.get_cut_points(clip[0], clip[1], n)
                clips = [(cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]
            labels = [label] * len(clips) if label == 0 else [label] + [label + 1] * (len(clips) - 1)
            for k, (clip, label) in enumerate(zip(clips, labels)):
                # if label == 2: clip = (clip[0] - 0.5 * fps, clip[1] - 0.5 * fps)
                video_name = name.replace("/", "_").split(".")[0] + "_{:0=3d}_{}.mp4".format(count, label)
                # video_name = os.path.join("video", video_name)
                video_name = os.path.join("video", str(label), video_name)
                save_video = os.path.join(save_dir, video_name)
                temp_file = "./temp.mp4"
                video_utils.video_capture(file, save_video=temp_file, vis=False, clip=clip, title="frame", delay=5)
                video_utils.video2video(temp_file, save_video, save_fps=30)
                count += 1
                outs_data.append({"file": video_name, "label": label})
        # if len(outs_data) >= 20: exit(0)
    json_utils.save_json(os.path.join(save_dir, "video_time_dataset.json"), outs_data)
    return outs_data


if __name__ == '__main__':
    anno_file = "/home/PKing/nasdata/tmp/tmp/fall/videos/Multiple-Cameras-Fall-Dataset/data_tuple3.csv"
    data_dir = "/home/PKing/nasdata/tmp/tmp/fall/videos/Multiple-Cameras-Fall-Dataset/dataset"
    save_dir = "/home/PKing/nasdata/tmp/tmp/fall/videos/Multiple-Cameras-Fall-Dataset/dataset-label"
    # video_info = clip_label_dataset(anno_file, data_dir, save_dir=save_dir)
    # get_ori_data_label(anno_file, data_dir)
    # TODO 处理正常视频
    anno_file = "/home/PKing/nasdata/tmp/tmp/fall/videos/Multiple-Cameras-Fall-Dataset/原始视频/others-fall-lie.json"
    save_dir = "/home/PKing/nasdata/tmp/tmp/fall/videos/Multiple-Cameras-Fall-Dataset/dataset-video-tmps"
    clip_time_video(anno_file, save_dir, chute="chute15")
    # file1 = "/home/PKing/Videos/cam1.avi"
    # file2 = "/home/PKing/Videos/cam2.avi"
    # video_utils.video2video(file1, file2, save_fps=30)
