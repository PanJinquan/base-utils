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
from pybaseutils.dataloader import parser_labelme, parser_yolo
from pybaseutils import file_utils, image_utils


def convert_labelme2yolo(anno_dir, out_root=None, class_name=None, use_seg=False, max_num=-1, vis=True, **kwargs):
    """
    将labelme格式转换为YOLO数据格式
    :param anno_dir:  输入labelme根目录(图片和json文件必须放在同一目录)
    :param out_root:  输出YOLO数据格根目录
    :param class_name: 需要选择的类别，None表示全部
    :param use_seg: 数据格式，True是YOLO实例分割数据格式 [class_index, cx, cy, w,  h]
                            False是YOLO目标检测格式 [class_index, x1, y1, x2, y2, x3, y3, x4, y4,....]
    :param max_num: 最多转换样本个数
    :param vis: 可视化标注效果
    """
    if anno_dir and not out_root: out_root = os.path.join(os.path.dirname(anno_dir), "yolo")
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
        labels, points, boxes = data_info["labels"], data_info["points"], data_info["boxes"]
        image, image_file = data_info["image"], data_info["image_file"]
        parser_yolo.save_yolo(out_root, image_file=image_file, boxes=boxes, points=points, labels=labels,
                              use_seg=use_seg, image=image, vis=vis)
    file_utils.write_data(os.path.join(out_root, "class_name.txt"), dataset.class_name)


if __name__ == "__main__":
    class_name = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish', 'Gourami', 'MorishIdol',
                  'PlatyFish', 'RibbonedSweetlips', 'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang',
                  'ZebraFish']
    anno_dir = "/home/PKing/nasdata/tmp/tmp/Fish/test/labelme/images"
    convert_labelme2yolo(anno_dir, class_name=class_name, use_seg=False, vis=False)
