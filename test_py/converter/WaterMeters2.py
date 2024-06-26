# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.converter import build_labelme
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc


def load_annotation(image_file, lable_file):
    image = image_utils.read_image(image_file)
    annos = file_utils.read_data(lable_file, split=" ", convertNum=False)
    annos = annos[0]
    point = [int(i) for i in annos[0:8]]
    point = np.asarray(point).reshape(-1, 2)
    label = annos[8]
    boxes = image_utils.polygons2boxes([point])
    return image, point, label, boxes


def WaterMeters(image_dir, label_dir, out_json, crop_root=None, use_align=False, vis=True, delay=10):
    if crop_root: file_utils.create_dir(crop_root)
    if out_json: file_utils.create_dir(out_json)
    file_list = file_utils.get_files_lists(image_dir, sub=True)
    for i, image_name in tqdm(enumerate(file_list)):
        postfix = image_name.split(".")[-1]
        lable_name = image_name.replace(f".{postfix}", ".txt")
        json_name = image_name.replace(f".{postfix}", ".json")
        image_file = os.path.join(image_dir, image_name)
        lable_file = os.path.join(label_dir, lable_name)
        image, point, label, boxes = load_annotation(image_file, lable_file)
        image_h, image_w = image.shape[:2]
        if crop_root and use_align:
            dst_pts = transform_utils.get_target_points(point)
            crop, M, Minv = transform_utils.get_image_alignment(image, src_pts=point, dst_pts=dst_pts, dsize=None,
                                                                method="lstsq")
            crop_file = os.path.join(crop_root, "{}_{:0=5d}_alignment.jpg".format(label, i))
            cv2.imwrite(crop_file, crop)
        elif crop_root:
            crop = image_utils.get_box_crop(image, box=boxes[0])
            crop_file = os.path.join(crop_root, "{}_{:0=5d}_crop.jpg".format(label, i))
            h, w = crop.shape[:2]
            if w > 3 * h: cv2.imwrite(crop_file, crop)
        json_file = os.path.join(out_json, json_name)
        build_labelme.maker_labelme(json_file, points=[point], labels=[label],
                                    image_name=image_name, image_size=(image_w, image_h))
        if vis:
            image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name=[label])
            image = image_utils.draw_key_point_in_image(image, [point], vis_id=True)
            # image_utils.cv_show_image("crop", crop, delay=10)
            image_utils.cv_show_image("image", image, delay=delay)
    return


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/tmp/tmp/水表数字识别/水表数据集/Water-Meter-Det1/val/images"
    label_dir = "/home/PKing/nasdata/tmp/tmp/水表数字识别/水表数据集/Water-Meter-Det1/val/labels"
    crop_root = "/home/PKing/nasdata/tmp/tmp/水表数字识别/水表数据集/Water-Meter-Det1/val/crop"
    out_json = "/home/PKing/nasdata/tmp/tmp/水表数字识别/水表数据集/Water-Meter-Det1/val/json"
    WaterMeters(image_dir, label_dir, out_json=out_json, crop_root=None, vis=True)
