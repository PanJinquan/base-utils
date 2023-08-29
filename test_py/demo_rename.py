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
    file_list = file_utils.get_images_list(image_dir)
    time = file_utils.get_time()
    file_utils.create_dir(output)
    for i, old in tqdm(enumerate(file_list)):
        postfix = old.split(".")[-1]
        name = "{}_{}".format(prefix, time) if prefix else time
        name = "{}_{:0=4d}.{}".format(name, i, postfix)
        new = os.path.join(output, name)
        file_utils.copy_file(old, new)


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/training_set/鹰"
    output = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/training_set/鹰-new"
    rename_image_file(image_dir, output, prefix="")
