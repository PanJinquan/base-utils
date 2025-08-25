# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-08 14:10:15
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils

import os
from pybaseutils.dataloader import parser_coco_kps

import os
from pybaseutils.dataloader import parser_voc

if __name__ == "__main__":
    # 修改为自己数据集的路径
    data_root = "/home/PKing/nasdata/tmp/tmp/pen/笔尖指尖标注方法"
    class_name = []
    dataset = parser_voc.VOCDataset(filename=None,
                                    data_root=data_root,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=False,
                                    check=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        print(image_id)
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        parser_voc.show_target_image(image, bboxes, labels, normal=False, transpose=False,
                                     class_name=class_name, use_rgb=False, thickness=3, fontScale=1.2)