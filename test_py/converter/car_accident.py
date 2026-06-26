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
from collections import defaultdict
from pybaseutils import json_utils, pandas_utils, image_utils, file_utils, dict_utils
from pybaseutils.cvutils import video_utils

class_maps = {
    "before crush": {"color": (255, 0, 0), "name": "crush0", "shift": [0, -1.0]},
    "crush moment": {"color": (0, 255, 0), "name": "crush1", "shift": [-1.0, 1.0]},
    "after crush": {"color": (0, 0, 255), "name": "crush2", "shift": [-1.0, 0]},
}


def parser_video(video_file, csv_file, vis=True):
    print("video_file", video_file)
    print(" csv_file", csv_file)
    pd_data = pandas_utils.read_csv(csv_file, sep=",")
    pd_dict = pandas_utils.df2dict(pd_data, orient="index")
    label_info = defaultdict(list)
    video_cap = video_utils.get_video_capture(video_file)
    width, height, num_frames, fps = video_utils.get_video_info(video_cap)
    for info in pd_dict.values():
        index = info['frame_num']
        label = info['type'].strip()
        label_info[label].append(index)
        if not vis: continue
        # 设置抽帧的位置
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video_cap.read()
        if not ret: break
        color = class_maps[label]["color"]
        text = "{:3d}-{}".format(index, label)
        frame = image_utils.draw_text(frame, (0, 0), text, color=color, fontScale=1, thickness=2)
        image_utils.show_image("image", frame, delay=20)
    return label_info, fps, num_frames


def segment_video_dir(video_dir, csv_dir, output="", vis=False):
    video_files = file_utils.get_files_list(video_dir, postfix=file_utils.VIDEO_POSTFIX)
    for i, video_file in enumerate(video_files):
        if not output: output = os.path.dirname(video_file)
        name, ext = os.path.splitext(os.path.basename(video_file))
        csv_file = os.path.join(csv_dir, name.replace("-", "_") + ".csv")
        if not os.path.exists(csv_file):
            print("not exists:{}".format(csv_file))
            continue
        label_info, fps, num_frames = parser_video(video_file, csv_file, vis=vis)
        for label, indices in label_info.items():
            clip = [min(indices), max(indices)]
            name = class_maps[label]["name"]
            shift = class_maps[label]["shift"]
            clip_index = [clip[0] + int(shift[0] * fps), clip[1] + int(shift[1] * fps)]
            clip_index = [max(0, clip_index[0]), min(num_frames, clip_index[1])]
            if (clip_index[1] - clip_index[0]) / fps < 0.5: continue
            nid, postfix = file_utils.split_postfix(video_file)
            # save_file = os.path.join(output, f"{nid}_{name}_crop{clip[0]}_{clip[1]}.mp4")
            save_file = os.path.join(output, name, f"video{i:03d}_{name}.mp4")
            video_utils.segment_video(video_file, save_file=save_file, clip_index=clip_index)


if __name__ == '__main__':
    video_dir = '/media/PKing/dev1/project/Crash dataset/video/'
    csv_dir = "/media/PKing/dev1/project/Crash dataset/label/csv"
    output = '/media/PKing/dev1/project/Crash dataset/video-crop/'
    segment_video_dir(video_dir, csv_dir, output=output)
