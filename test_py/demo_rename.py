# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-25 17:42:55
    @Brief  :
"""
import os
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


def rename_image_file(image_dir, output, prefix=""):
    file_list = file_utils.get_files_lists(image_dir)
    time = file_utils.get_time()
    file_utils.create_dir(output)
    for i, old in tqdm(enumerate(file_list)):
        postfix = old.split(".")[-1]
        name = "{}_{}".format(prefix, time) if prefix else time
        name = "{}_{:0=4d}.{}".format(name, i, postfix)
        new = os.path.join(output, name)
        file_utils.copy_file(old, new)


if __name__ == '__main__':

    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/使用钳形电流表测量低压线路电流/天河数据/2024-04-12-1-2-train"
    output = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/使用钳形电流表测量低压线路电流/天河数据/2024-04-12-1-2-train2"
    rename_image_file(image_dir, output, prefix="南沙")
