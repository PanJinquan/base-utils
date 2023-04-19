# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-17 18:43:21
    @Brief  :
"""
from pybaseutils import file_utils, image_utils

if __name__ == "__main__":
    """深度学习目标检测"""
    image_dir = "/home/dm/nasdata/Detector/YOLO/yolov5/runs/test_image-result"
    gif_file = image_dir + ".gif"
    frames = file_utils.get_images_list(image_dir)
    # image_utils.image_file2gif(frames, size=(416, None), padding=False,interval=1, gif_file=gif_file, fps=1)
    image_utils.image_file2gif(frames, size=(640, 640), padding=True, interval=1,
                               gif_file=gif_file, fps=1, use_pil=True)
