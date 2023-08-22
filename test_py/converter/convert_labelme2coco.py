# -*-coding: utf-8 -*-
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
from pybaseutils.converter.convert_labelme2coco import Labelme2COCO


def demo_for_voc():
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    save_coco_file = os.path.join(os.path.dirname(image_dir), "person.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    build.build_keypoint_dataset()
    build.save_coco(save_coco_file)


if __name__ == '__main__':
    demo_for_voc()
