# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-03-18 16:16:45
    @Brief  :
"""

# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_voc
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils import file_utils, image_utils

VOC_NAMES = ["BG", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
             'tvmonitor']


def convert_voc2labelme(image_dir,
                        anno_root=None,
                        out_root=None,
                        class_dict=None,
                        vis=True):
    """
    将VOC格式转换为labelme格式，以便重新label重新映射
    :param filename:
    :param out_xml_dir: output VOC XML,Annotations
    :param out_img_dir: output VOC image if not None ,JPEGImages
    :param class_dict:
    """
    target_label = [VOC_NAMES.index(n) for n in class_dict.keys()]
    image_files = file_utils.get_files_lists(image_dir)
    for image_file in tqdm(image_files):
        # image_file = "/home/PKing/nasdata/tmp/face_person/VOCdevkit/SBD/JPEGImages/2008_000026.jpg"
        image_name = os.path.basename(image_file)
        annos_file = os.path.join(anno_root, image_name.replace(".jpg", ".png"))
        image = cv2.imread(image_file)
        masks = cv2.imread(annos_file, flags=cv2.IMREAD_UNCHANGED)
        labels = list(set(masks.reshape(-1).astype(np.int32)))
        labels = [n for n in labels if n > 0]
        names = [VOC_NAMES[i] for i in labels if i in target_label]
        points = image_utils.find_image_contours(masks, target_label=target_label)
        points = [p[0] for p in points if p]
        if len(points) > 0 and out_root:
            h, w = image.shape[:2]
            image_id = image_name.split(".")[0]
            json_file = os.path.join(out_root, "images", f"{image_id}.json")
            img_file = os.path.join(out_root, "images", image_name)
            file_utils.copy_file(image_file, img_file)
            build_labelme.maker_labelme(json_file, points, names, image_name, image_size=[w, h], image_bs64=None)


if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/tmp/face_person/VOCdevkit/SBD/JPEGImages"
    anno_root = "/home/PKing/nasdata/tmp/face_person/VOCdevkit/SBD/SegmentationClass"
    out_root = "/home/PKing/nasdata/tmp/face_person/VOCdevkit/SBD/labelme"
    class_dict = {"person": "未穿工作服"}
    convert_voc2labelme(image_dir, anno_root=anno_root, out_root=out_root, class_dict=class_dict)
