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
    """
    v2:{'绝缘鞋': 1, '身穿工作服': 2, '手': 3, '螺丝刀': 4, '验电笔': 5, '准备区域': 6, '钳形电流表': 7, '万用表': 8, '相序表': 9, '护目镜': 10, '绝缘手套': 11, '竹梯': 12, '表箱关': 13, '安全带': 14, '工作台': 15, '未穿工作服': 16, '其他鞋': 17, '相序表线头': 18, '万用表线头': 19, '电能表': 20, '电流互感器': 21, '接线盒': 22, '安全帽': 23}
    v4:{'电流互感器': 1, '表箱开': 2, '表箱关': 3, '竹梯': 4, '接线盒': 5, '电能表': 6, '安全帽': 7, '护目镜': 8, '绝缘手套': 9, '身穿工作服': 10, '工作台': 11, '准备区域': 12, '安全带': 13, '其他鞋': 14, '未穿工作服': 15, '万用表': 16, '螺丝刀': 17, '验电笔': 18, '绝缘鞋': 19, '万用表线头': 20, '相序表': 21, '相序表线头': 22, '钳形电流表': 23, '手': 24}
    :return:
    """
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v4/train.txt"
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v4/JPEGImages"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v4/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "train_coco_instance.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, filename=filename, init_id=None)
    build.build_instances_dataset(coco_ins_file)


if __name__ == '__main__':
    # demo_for_person5()
    demo_for_aije()
