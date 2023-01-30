# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import threading
import time
import torch
from tqdm import tqdm
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pybaseutils import file_utils, image_utils

# table = {"竖画平行": "同竖平行", "撇画平行": "同撇平行"}
table = {"竖画平行": "同竖平行", "横画平行": "同横平行", "撇画平行": "同撇平行"}


def rename_image_dir(image_dir, table=table):
    image_list = file_utils.get_images_list(image_dir)
    for src_file in tqdm(image_list):
        name = os.path.basename(src_file)
        for k, v in table.items():
            if k not in name:
                continue
            dst_file = src_file.replace(k, v)
            if os.path.exists(dst_file):
                # print("exists:{}".format(dst_file))
                continue
            print(os.path.dirname(src_file), "{}->{}".format(os.path.basename(src_file), os.path.basename(dst_file)))
            # file_utils.copy_file(src_file, dst_file)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/release/handwriting/calligraphy-evaluation/FontLibrary/font-library-v6/楷体GB2312/struct_image"
    rename_image_dir(image_dir, )
