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
from pybaseutils.dataloader import parser_yolo
from pybaseutils import file_utils, image_utils


def get_multi_obj_kpts_info(data_info):
    """
    多目标关键点
    :param data_info:
    :return:
    """
    names_, boxes = data_info["names"], data_info["boxes"]
    points = image_utils.boxes2polygons(boxes).tolist()
    groups = list(range(1, len(points) + 1))
    assert len(names_) == len(points)
    for gid, kpts in enumerate(data_info["points"]):  # 关键点
        for pid, kpt in enumerate(kpts):
            if kpt[2] > 0:
                points.append([kpt[0:2].tolist()])
                names_.append(f"p{pid}")
                groups.append(groups[gid])  # 实例ID(组ID),相同目标和关键点使用同一个组ID
    return points, names_, groups


def get_single_obj_kpts_info(data_info):
    """
    单目标关键点
    :param data_info:
    :return:
    """
    class_dict = {'pressure_meter': 'pressure_meter',
                  'pointer#0': 'pointer_start',
                  'pointer#1': 'pointer_end',
                  'range_start#0': 'range_start',
                  'range_end#0': 'range_end'
                  }
    image = data_info["image"]
    h, w = image.shape[:2]
    boxes = [[0, 0, w, h]]
    boxes = image_utils.extend_xyxy(boxes, scale=(0.95, 0.95))
    points = image_utils.boxes2polygons(boxes).tolist()
    names_ = ["pressure_meter"]
    target = data_info["names"]
    groups = [1]
    for gid, kpts in enumerate(data_info["points"]):  # 关键点
        for pid, kpt in enumerate(kpts):
            if kpt[2] > 0:
                points.append([kpt[0:2].tolist()])
                names_.append(f"{target[gid]}#{pid}")
                groups.append(1)
    try:
        names_ = [class_dict[n] for n in names_]
    except:
        points, names_, groups = [], [], []
    return points, names_, groups


def convert_yolo2labelme(data_root,
                         out_root=None,
                         class_name=None,
                         class_dict={},
                         task="",
                         check=False,
                         prefix="",
                         max_num=-1,
                         vis=True,
                         **kwargs):
    """
    将labelme格式转换为labelme格式
    :param data_root:  输入labelme根目录
    :param out_root:  输出labelme根目录
    :param class_name: 需要选择的类别，None表示全部
    :param class_dict: 类别映射
    :param prefix: 提供文件名前缀，则进行重新命名
    :param max_num: 最多转换样本个数
    """
    if data_root and not out_root: out_root = os.path.join(data_root, "labelme")
    dataset = parser_yolo.YOLODataset(filename=None,
                                      data_root=data_root,
                                      anno_dir=None,
                                      image_dir=None,
                                      class_name=class_name,
                                      task=task,
                                      check=check,
                                      phase="val",
                                      shuffle=False)
    print("have num:{}".format(len(dataset)))
    nums = min(len(dataset), max_num) if max_num > 0 else len(dataset)
    for i in tqdm(range(nums)):
        data_info = dataset.__getitem__(i)
        image, names, points, boxes = data_info["image"], data_info["names"], data_info["points"], data_info["boxes"]
        image_file = data_info["image_file"]
        group = None
        if task == "pose":
            # points, names, group = get_multi_obj_kpts_info(data_info)
            points, names, group = get_single_obj_kpts_info(data_info)
        build_labelme.save_labelme(out_root, image_file=image_file, points=points, names=names,
                                   class_dict=class_dict, group=group, image=image, prefix=prefix, index=i, vis=vis)
    file_utils.write_data(os.path.join(out_root, "class_name.txt"), dataset.class_name)


if __name__ == "__main__":
    class_name = ['pointer', 'range_start', 'range_end']
    data_root = "/home/PKing/nasdata/dataset/指针表计/dataset/指针仪表数据集/关键点检测(YoloV8Pose)/data_pose/train"
    # data_root = "/home/PKing/nasdata/dataset/指针表计/dataset/指针仪表数据集/关键点检测(YoloV8Pose)/data_pose/val"
    convert_yolo2labelme(data_root, class_name=class_name, prefix="train", task="pose", vis=False)
