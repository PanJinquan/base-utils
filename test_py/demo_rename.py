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
    # image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v12/黄埔数企"
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/train-v2/aije-action-train-v25/JPEGImages"
    output = image_dir + "-train"
    prefix = "江门_更换熔丝"
    # image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-cls2/手与表箱门有接触/手与表箱门无接触/南沙工匠"
    # output = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/室内考题/南沙视频/室内/南沙基地视频3/南沙工匠"
    # prefix = os.path.basename(output)
    rename_image_file(image_dir, output, prefix=prefix)
