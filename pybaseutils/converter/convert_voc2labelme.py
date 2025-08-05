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
from pybaseutils.dataloader import parser_voc
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils import file_utils, image_utils


def convert_voc2labelme(filename=None,
                        data_root=None,
                        out_root=None,
                        class_name=None,
                        class_dict={},
                        prefix="",
                        max_num=-1,
                        vis=True):
    """
    将voc格式转换为labelme格式
    :param anno_dir:  输入labelme根目录
    :param out_root:  输出labelme根目录
    :param class_name: 需要选择的类别，None表示全部
    :param class_dict: 类别映射
    :param prefix: 提供文件名前缀，则重新进行重新命令
    """
    if data_root and not out_root: out_root = os.path.join(data_root, "labelme")
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=data_root,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=False,
                                    check=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    nums = min(len(dataset), max_num) if max_num > 0 else len(dataset)
    for i in tqdm(range(nums)):
        data_info = dataset.__getitem__(i)
        image, boxes, label = data_info["image"], data_info["boxes"], data_info["labels"]
        label = np.asarray(label, np.int32).reshape(-1).tolist()
        names = [dataset.class_name[i] for i in label]
        image_file = data_info["image_file"]
        points = image_utils.boxes2polygons(boxes)
        build_labelme.save_labelme(out_root, image_file=image_file, points=points, names=names,
                                   class_dict=class_dict, image=image, prefix=prefix, index=i, vis=vis)


if __name__ == "__main__":
    # data_root = "/home/PKing/nasdata/dataset/face_person/VOC/VOC2007"
    # data_root = "/home/PKing/nasdata/dataset/face_person/VOC/VOC2012"
    # data_root = "/home/PKing/nasdata/dataset/face_person/MPII"
    data_root = "/home/PKing/nasdata/dataset/face_person/COCO"
    out_root = "/home/PKing/nasdata/dataset/face_person/labelme/{}".format(os.path.basename(data_root))
    class_name = None
    convert_voc2labelme(data_root=data_root, out_root=out_root, class_name=class_name, vis=False)
