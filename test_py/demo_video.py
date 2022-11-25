# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-24 14:03:56
    @Brief  :
"""
from pybaseutils import image_utils
from pybaseutils.cvutils import video_utils


def crop_fun(frame):
    """1920 × 1080"""
    h, w = frame.shape[:2]
    frame = image_utils.get_bbox_crop(frame, bbox=[50, 30, 1400, 1080])
    return frame


def resize_fun(frame):
    h, w = frame.shape[:2]
    frame = image_utils.resize_image(frame, size=(640, None))
    return frame


if __name__ == "__main__":
    # video_file = "/home/dm/nasdata/dataset/csdn/文档矫正/文档矫正效果.mp4"
    # video_file = "/home/dm/nasdata/Demo/鼠标绘图/Kazam_screencast_00004.mp4"
    # video_file = "/home/dm/视频/双目测距Demo视频(Python).mp4"
    # dst_file = "/home/dm/视频/双目测距Demo视频(Python)1.mp4"
    # video_utils.video2frames(video_file, interval=2, func=crop_fun, vis=True)
    # video_utils.video2gif(video_file, interval=2, func=target_task, fps=3, use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)

    # image_dir = "/home/dm/nasdata/dataset/csdn/文档矫正/文档矫正效果2"
    # video_utils.frames2video(image_dir, interval=1, vis=True)
    video_file = "/home/dm/nasdata/dataset/csdn/文档矫正/文档矫正效果2_20221124_170344_8152.mp4"
    video_utils.video2gif(video_file, interval=6, func=resize_fun, fps=3, use_pil=False, vis=True)
    # image_utils.image_file_list2gif(image_dir)
