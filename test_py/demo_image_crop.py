# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-03-07 14:36:25
    @Brief  :
"""
import os
import cv2
from pybaseutils import file_utils, image_utils


def crop_images(image_dir, out_dir, box=[], vis=True):
    files = file_utils.get_files_lists(image_dir)
    for image_file in files:
        out_file = os.path.join(out_dir, os.path.basename(image_file))
        image = image_utils.read_image(image_file)
        h, w = image.shape[:2]  # (1080,1920)
        bbox = box if len(box) == 4 else [0, 0, w, h]
        crop = image_utils.get_box_crop(image, box=bbox)
        if out_file:
            cv2.imwrite(out_file, crop)
        if vis:
            image_utils.cv_show_image("image", image, delay=10)
            image_utils.cv_show_image("crop", crop)


if __name__ == "__main__":
    """批量裁剪图片"""
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/专利图/images"
    out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/专利图/crops"
    x1 = 450
    y1 = 1080 - 700
    box = [x1, y1, x1 + 900, 1080]
    crop_images(image_dir, out_dir=out_dir, box=box, vis=True)
