# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-24 14:03:56
    @Brief  :
"""
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import video_utils


def crop_fun(frame):
    """1920 × 1080"""
    h, w = frame.shape[:2]
    frame = image_utils.get_bbox_crop(frame, bbox=[60, 5, 1280, 815])
    return frame


def resize_fun(frame):
    h, w = frame.shape[:2]
    frame = image_utils.resize_image(frame, size=(None, 256))
    return frame


if __name__ == "__main__":
    video_file = "/home/dm/nasdata/dataset/csdn/plate/CCPD-master/CRNN-Plate-Recognition/docs/result2.avi"
    # video_utils.video2frames(video_file, interval=2, func=None, vis=True)
    video_utils.video2gif(video_file, interval=8, func=resize_fun, fps=3, use_pil=False, vis=True)
    # video_utils.video2gif(video_file, interval=4, func=None, fps=5, use_pil=True, vis=True)
    # video2video(video_file, dst_file, vis=True)

    # image_dir = "/home/dm/视频/Kazam_screencast_00000"
    # video_utils.frames2video(image_dir,interval=1, fps=1)
    # image_utils.image_dir2gif(image_dir, size=(None, 600), interval=6, fps=2, loop=0, padding=False, use_pil=False)
