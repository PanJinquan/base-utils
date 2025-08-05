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
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.converter import build_labelme
from pybaseutils.dataloader import parser_labelme
from pybaseutils import file_utils, image_utils


def convert_labelme2labelme(anno_dir,
                            out_root=None,
                            class_name=None,
                            class_dict={},
                            prefix="",
                            max_num=-1,
                            vis=True,
                            **kwargs):
    """
    将labelme格式转换为labelme格式
    :param anno_dir:  输入labelme根目录
    :param out_root:  输出labelme根目录
    :param class_name: 需要选择的类别，None表示全部
    :param class_dict: 类别映射
    :param prefix: 提供文件名前缀，则进行重新命名
    :param max_num: 最多转换样本个数
    """
    if anno_dir and not out_root: out_root = os.path.join(os.path.dirname(anno_dir), "labelme")
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=class_name,
                                             check=False,
                                             phase="val",
                                             shuffle=False)
    print("have num:{}".format(len(dataset)))
    nums = min(len(dataset), max_num) if max_num > 0 else len(dataset)
    for i in tqdm(range(nums)):
        data_info = dataset.__getitem__(i)
        image, names, points = data_info["image"], data_info["names"], data_info["points"]
        image_file = data_info["image_file"]
        build_labelme.save_labelme(out_root, image_file=image_file, points=points, names=names,
                                   class_dict=class_dict, image=image, prefix=prefix, index=i, vis=vis)


if __name__ == "__main__":
    anno_file = "/home/PKing/nasdata/tmp/tmp/cat-det/dataset/video/car-dataset"
    out_root = "/home/PKing/nasdata/tmp/tmp/cat-det/dataset/video/labelme"
    convert_labelme2labelme(anno_file, out_root, class_name=[], vis=False)
