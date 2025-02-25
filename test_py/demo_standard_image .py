# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils


def image_clip_demo(image_dir, out_dir):
    """
    显示图片大小
    pip install pybaseutils==0.7.3
    :param image_dir: 输入图片文件夹
    :param out_dir:保存图片文件夹
    :return:
    """
    image_list = file_utils.get_images_list(image_dir)
    for image_file in image_list:
        image = cv2.imread(image_file)
        # 限制图像最大分辨率,不超过1920
        out_image = image_utils.resize_image_clip(image, clip_max=1920)
        print(image.shape, out_image.shape)
        if out_dir:
            # out_file = file_utils.create_dir(out_dir, None, os.path.basename(image_file))  # 原始格式
            # out_file = file_utils.create_dir(out_dir, None, f"{os.path.basename(image_file).split('.')}.jpg") # 转为jpg格式
            out_file = file_utils.create_dir(out_dir, None, f"image_{file_utils.get_time()}.jpg") # 转为jpg格式
            cv2.imwrite(out_file, out_image)


if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-date/date-det/dataset-v01/原始数据"
    out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-date/date-det/dataset-v01/images"
    image_clip_demo(image_dir, out_dir)
