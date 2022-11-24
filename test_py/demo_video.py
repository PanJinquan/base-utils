# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-24 14:03:56
    @Brief  :
"""
from pybaseutils import video_utils, image_utils


def crop_fun(frame):
    h, w = frame.shape[:2]
    frame = image_utils.get_bbox_crop(frame, bbox=[w // 2 + 130, 0, w, h])
    return frame


def resize_fun(frame):
    h, w = frame.shape[:2]
    frame = image_utils.resize_image(frame, size=(480, None))
    return frame


if __name__ == "__main__":
    # video_file = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00001.mp4"
    # video_file = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00004.mp4"
    # video_file = "/home/dm/视频/双目测距Demo视频(Python).mp4"
    # dst_file = "/home/dm/视频/双目测距Demo视频(Python)1.mp4"
    # video_utils.video2frames(video_file, interval=2, func=target_task, vis=True)
    # frames2video(image_dir, interval=1, vis=True)
    # video_utils.video2gif(video_file, interval=2, func=target_task, fps=3, use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)

    # image_dir = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00004"
    # video_utils.frames2video(image_dir, interval=1, vis=True)
    # video_file = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00001_20221124_142847_6257.mp4"
    video_file = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00004_20221124_142902_8286.mp4"
    video_utils.video2gif(video_file, interval=2, func=resize_fun, fps=10, use_pil=False, vis=True)

    # image_utils.image_file_list2gif(image_dir)
