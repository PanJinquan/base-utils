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
    frame = image_utils.resize_image(frame, size=(256, None))
    return frame


if __name__ == "__main__":
    video_file = "/home/dm/nasdata/dataset/csdn/traffic light/demo/test-video-result.mp4"
    # video_utils.video2frames(video_file, interval=5, func=crop_fun, vis=True)
    video_utils.video2gif(video_file, interval=7, func=resize_fun, fps=5, use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)

    # image_dir = "/home/dm/nasdata/Detector/YOLO/yolov5/data/test_image"
    # video_utils.frames2video(image_dir,interval=1, fps=1)
    # image_utils.image_dir2gif(image_dir, size=(None, 720), interval=5, fps=4, loop=0, padding=False, use_pil=True)
