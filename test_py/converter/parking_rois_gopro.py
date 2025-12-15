# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-12-10 15:21:26
# @Brief  : https://opendatalab.org.cn/OpenDataLab/Action-Camera_Parking/tree/main/raw
# --------------------------------------------------------
"""
import os
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, json_utils, image_utils
from pybaseutils.converter import build_labelme


def convert2labelme(json_file, out_root, vis=False):
    root = os.path.dirname(json_file)
    image_dir = os.path.join(root, 'images')
    src_data = json_utils.load_json(json_file)
    class_list = ['empty', 'occupied']
    for phase, data in tqdm(src_data.items()):
        files_ = data['file_names']
        points = data['rois_list']
        labels = data['occupancy_list']
        for name, point, label in zip(files_, points, labels):
            file = os.path.join(image_dir, name)
            image = image_utils.read_image(file, size=(None, 1080))
            h, w = image.shape[:2]
            point = np.asarray(point).astype(np.float32) * (w, h)
            label = np.asarray(label).astype(np.int32)
            names = [class_list[i] for i in label]
            build_labelme.save_labelme(out_root, image_file=file, image=image, points=point, names=names,
                                       prefix="image", index=-1)
            if vis:
                image = image_utils.draw_image_contours(image, point, names)
                image_utils.show_image("image", image)


if __name__ == '__main__':
    json_file = "/home/PKing/nasdata/tmp/tmp/Parking/dataset/parking_rois_gopro/annotations.json"
    out_root = "/home/PKing/nasdata/tmp/tmp/Parking/dataset/parking_rois_gopro/labelme"
    convert2labelme(json_file, out_root)
