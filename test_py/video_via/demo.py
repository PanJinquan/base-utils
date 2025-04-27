# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-04-15 19:51:59
# @Brief  :
# --------------------------------------------------------
"""
import pandas as pd
from pybaseutils import json_utils, pandas_utils, image_utils
from pybaseutils.cvutils import video_utils


def read_csv(filename, sep=","):
    """
    :param filename:
    :param sep: 分隔符
    :return:
    """
    names = ["name", "file_list", "temporal_segment_start", "temporal_segment_end", "metadata"]
    file = pd.read_csv(filename, sep=sep, names=names, comment="#")
    df = pd.DataFrame(file)
    return df


def load_annotation(filename):
    """
    :param filename:
    :return:
    """
    df = read_csv(filename)
    label = df['metadata'].tolist()
    times = df[['temporal_segment_start', 'temporal_segment_end']].values.tolist()
    label = [json_utils.str2dict(s)["TEMPORAL-SEGMENTS"] for s in label]
    assert len(label) == len(times), f"数据标注有问题：{filename}"
    return label, times


def get_video_label(labels, times, count, fps, offset=0):
    t = count / fps
    c = -1
    for i, clip in enumerate(times):
        t0 = clip[0] + offset
        t1 = clip[1] - offset
        if t0 < t < t1: c = i
    label = labels[c] if c >= 0 else "face"
    return label


def parser_video(video_file, annot_file):
    video = video_utils.video_iterator(video_file, save_video=None, vis=True, delay=50)
    labels, times = load_annotation(annot_file)
    for data_info in video:
        count = data_info["count"]
        frame = data_info["frame"]
        label = get_video_label(labels, times, count, fps=data_info['fps'])
        frame = image_utils.draw_text(frame, point=(10, 50), text=label, drawType="chinese")
        data_info["frame"] = frame


if __name__ == '__main__':
    video_file = "/media/PKing/新加卷1/个人文件/video/driving/DF0001.mp4"
    annot_file = "/media/PKing/新加卷1/个人文件/video/driving/DF0001.csv"
    parser_video(video_file, annot_file)
