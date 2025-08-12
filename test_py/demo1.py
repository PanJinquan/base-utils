# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-22 10:40:45
# @Brief  :
# --------------------------------------------------------
"""
import os
from pybaseutils.dataloader import parser_labelme

if __name__ == "__main__":
    anno_dir = "/media/PKing/新加卷/SDK/base-utils/data/labelme/images"
    # names = ['person', 'car']
    names = ['car']
    # names = ['person']
    kpts_name = ['p0', 'p1', "p2", "p3", "p4", "p5"]
    kpts_name = ['p1', 'p0', "p2", "p3", "p4", "p5"]
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=names,
                                             use_kpts=True,
                                             kpts_name=kpts_name,
                                             check=True,
                                             phase="val",
                                             shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        # i = 5
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, points, boxes, labels = data["image"], data["points"], data["boxes"], data["labels"]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        annot_file = os.path.join("masker", "{}.json".format(os.path.basename(image_file).split(".")[0]))
        print(image_file)
        parser_labelme.show_target_image(image, boxes, labels, points, bones_type="", keypoints=data["keypoints"])
