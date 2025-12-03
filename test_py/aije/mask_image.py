# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os

import cv2
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.dataloader import parser_labelme


def get_mask_image(image_dir, mask_file, output=None, prefix="", color=(128, 128, 128), vis=False):
    """
    从标注文件中提取掩码信息，然后绘制掩码并显示在窗口中
    :param image_dir: 图像目录
    :param mask_file: 掩码文件(使用labelme格式)
    :return:
    """
    if output: os.makedirs(output, exist_ok=True)
    image_list = file_utils.get_files_list(image_dir, prefix=prefix, postfix=file_utils.IMG_POSTFIX)
    mask_info = parser_labelme.parser_labelme(mask_file, class_dict={}, size=())
    for image_file in tqdm(image_list):
        print(image_file)
        image = image_utils.read_image(image_file)
        points, bboxes, names = mask_info["points"], mask_info["boxes"], mask_info["names"]
        image = image_utils.draw_contours(image, points, color=color, thickness=-1)
        if output:
            outfile = os.path.join(output, os.path.basename(image_file))
            cv2.imwrite(outfile, image)
        if vis:
            image_utils.cv_show_image("image", image)
    return image_list


if __name__ == "__main__":
    """
    从标注文件中提取掩码信息，然后绘制掩码并显示在窗口中
    """
    image_dir = "/home/PKing/Pictures/DMovie"
    output = "/home/PKing/Pictures/DMovieMask"
    mask_file = "/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/被忽略的区域/images/image-2025-11-17-10h12m33s569.json"
    get_mask_image(image_dir, mask_file, output=output, prefix="*image6", color=(128, 128, 128), vis=True)
