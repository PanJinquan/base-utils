# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-17 18:43:21
    @Brief  :
"""
import cv2
from pybaseutils import file_utils, image_utils
from pybaseutils.cvutils import video_utils


def resize_fun(frame):
    h, w = frame.shape[:2]
    # frame = image_utils.resize_image(frame, size=(224, None))
    frame = image_utils.resize_image(frame, size=(480, None))
    frame = cv2.flip(frame)
    # frame = image_utils.resize_image(frame, size=(960, None))
    # frame = image_utils.resize_image(frame, size=(None, 640))  # android-video
    return frame


if __name__ == "__main__":
    image_dir = "/home/PKing/Pictures/DMovie/image"
    gif_file = image_dir + ".gif"
    image_utils.image_dir2gif(image_dir, size=(640, None), crop=[0, 200, 1500, 1080],
                              gif_file=gif_file, interval=1, fps=1, )
    # video_file = "/home/PKing/Downloads/f09aec8a8a03e73c7f86fe3f340d62f8.gif"
    # video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/方案图/时序分析/未开时序分析.mp4"
    # video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/方案图/时序分析/开启时序分析.mp4"
    # video_utils.video2gif(video_file, interval=2, use_pil=False, func=resize_fun, fps=10)
