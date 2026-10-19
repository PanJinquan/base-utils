# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
import argparse
import sys
import os
import numpy as np
import json
import cv2
import time
from tqdm import tqdm
from pybaseutils import file_utils, json_utils, image_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import convert_labelme2coco


def demo():
    class_name = ["pressure_meter"]
    kpt_names = [ [ "range_start", "range_end",'pointer_start', 'pointer_end', ] ]
    skeleton =  [[2, 0], [2, 1], [2, 3]]
    kpt_shape = [4, 3]
    anno_dir = "/home/PKing/nasdata/tmp/tmp/pressure_meter/dataset-v1/val/images"
    coco_ins_file = os.path.join(os.path.dirname(anno_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(anno_dir), "coco_kps.json")
    build = convert_labelme2coco.Labelme2COCO(image_dir=anno_dir,
                                              anno_dir=anno_dir,
                                              class_name=class_name,
                                              use_kpt=True,
                                              kpt_names=kpt_names,
                                              kpt_shape=kpt_shape
                                              )
    # TODO 构建关键点检测数据集
    build.build_keypoints_dataset(coco_kps_file, skeleton=skeleton)
    # TODO 构建实例分割数据集
    # build.build_instances_dataset(coco_ins_file)


if __name__ == '__main__':
    demo()
