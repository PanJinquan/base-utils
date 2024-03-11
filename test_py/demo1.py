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


def WaterMeters(image_dir, label_dir, out_root, use_align=False, vis=True):
    if out_root: file_utils.create_dir(out_root)
    file_list = file_utils.get_files_lists(image_dir, sub=True)
    contents = []
    for i, image_name in tqdm(enumerate(file_list)):
        postfix = image_name.split(".")[-1]
        lable_name = image_name.replace(f".{postfix}", ".txt")
        image_file = os.path.join(image_dir, image_name)
        lable_file = os.path.join(label_dir, "gt_" + lable_name)
        if os.path.exists(image_file) and os.path.exists(lable_file):
            item = file_utils.get_sub_list([image_file, lable_file], out_root)
            contents.append(item)
        else:
            print("not exists:{}".format(image_file))
    filename = os.path.join(out_root, "lable.txt")
    file_utils.write_data(filename, contents, split="\t")
    # image, point, label, boxes = load_annotation(image_file, lable_file)
    # if use_align:
    #     dst_pts = transform_utils.get_target_points(point)
    #     crop, M, Minv = transform_utils.image_alignment(image, src_pts=point, dst_pts=dst_pts, out_size=None,
    #                                                     method="lstsq")
    #     crop_file = os.path.join(out_root, "{}_{:0=5d}_alignment.jpg".format(label, i))
    #     cv2.imwrite(crop_file, crop)
    # else:
    #     crop = image_utils.get_bbox_crop(image, bbox=boxes[0])
    #     crop_file = os.path.join(out_root, "{}_{:0=5d}_crop.jpg".format(label, i))
    #     h, w = crop.shape[:2]
    #     if w > 3 * h: cv2.imwrite(crop_file, crop)
    # if vis:
    #     image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name=[label])
    #     image = image_utils.draw_key_point_in_image(image, [point], vis_id=True)
    #     image_utils.cv_show_image("crop", crop, delay=10)
    #     image_utils.cv_show_image("image", image)
    return


if __name__ == '__main__':
    pass
