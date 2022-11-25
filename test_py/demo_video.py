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
    frame = image_utils.get_bbox_crop(frame, bbox=[60, 5, 1280, 815])
    return frame


def resize_fun(frame):
    h, w = frame.shape[:2]
    frame = image_utils.resize_image(frame, size=(640, None))
    return frame


if __name__ == "__main__":
    video_file = "/home/dm/nasdata/dataset/csdn/文档矫正/video/document_correct_by_auto2.mp4"
    # video_utils.video2frames(video_file, interval=2, func=crop_fun, vis=True)
    # video_utils.video2gif(video_file, interval=2, func=target_task, fps=3, use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)

    image_dir = "/home/dm/nasdata/dataset/csdn/文档矫正/video/document_correct_by_auto2"
    image_utils.image_dir2gif(image_dir, size=(None, 720), interval=5, fps=4, loop=0, padding=False, use_pil=True)
