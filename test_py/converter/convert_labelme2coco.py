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


def demo_for_person5():
    kps_name = ["p0", "p1", "p2", "p3", "p4"]
    skeleton = [[0, 2], [2, 1], [2, 3], [3, 4]]
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    # build.build_keypoints_dataset(coco_kps_file,
    #                               class_name=["person"],
    #                               kps_name=kps_name,
    #                               skeleton=skeleton,
    #                               num_joints=5)
    build.build_instances_dataset(coco_ins_file)


def demo_for_hand21():
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    # build.build_keypoints_dataset(coco_kps_file, class_name=["person"], num_joints=5)
    build.build_instances_dataset(coco_ins_file)


def demo_for_aije():
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v1/train.txt"
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v1/JPEGImages"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v1/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "train_coco_instance.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, filename=filename, init_id=None)
    build.build_instances_dataset(coco_ins_file)


if __name__ == '__main__':
    # demo_for_person5()
    demo_for_aije()
