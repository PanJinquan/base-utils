# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-17 18:43:21
    @Brief  :
"""
from pybaseutils import file_utils, image_utils

if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/car/UA-DETRAC/DETRAC-train-data/result"
    gif_file = image_dir + ".gif"
    frames = file_utils.get_images_list(image_dir)
    image_utils.image_file2gif(frames, size=(416, None), padding=False, gif_file=gif_file, fps=20)
