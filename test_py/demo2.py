# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils, font_utils, json_utils

if __name__ == "__main__":
    json_dir = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/grid_cross_points_images/grid_cross_points_soft_v1/json"
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/grid_cross_points_images/grid_cross_points_soft_v1/images"
    outdir = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/background/bg3"
    image_files = file_utils.get_files_lists(image_dir)
    for image_file in image_files:
        basename = os.path.basename(image_file)
        json_file = os.path.join(json_dir, basename.replace(".jpg", ".json"))
        if not os.path.exists(json_file):
            dst_file = file_utils.create_dir(outdir, "images", basename)
            # file_utils.copy_file(image_file, dst_file)
            file_utils.move_file(image_file, dst_file)
