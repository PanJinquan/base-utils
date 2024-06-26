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
from pybaseutils import image_utils, file_utils, pandas_utils
from pybaseutils.converter import build_labelme
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc


def load_annotation(image_dir, anno_info):
    image_name = anno_info[0]
    label = anno_info[1]
    data = eval(anno_info[2])
    image_file = os.path.join(image_dir, image_name)
    image = cv2.imread(image_file)
    h, w = image.shape[:2]
    point = [[d['x'], d['y']] for d in data['data']]
    point = np.array(point) * (w, h)
    boxes = image_utils.polygons2boxes([point])
    label = "{:0=9.3f}".format(label).replace(".", "")
    # label = str(label)
    return image, image_name, point, label, boxes


def WaterMeters(image_dir, ann_file, out_json, crop_root=None, use_align=True, vis=True, delay=5):
    if crop_root: file_utils.create_dir(crop_root)
    if out_json: file_utils.create_dir(out_json)
    pd = pandas_utils.read_csv(anno_file)
    annotation = pd[["photo_name", "value", "location"]].values.tolist()
    for i, anno_info in tqdm(enumerate(annotation)):
        image, image_name, point, label, boxes = load_annotation(image_dir, anno_info)
        postfix = image_name.split(".")[-1]
        json_name = image_name.replace(f".{postfix}", ".json")
        image_h, image_w = image.shape[:2]
        if crop_root and use_align:
            src_pts = image_utils.find_minAreaRect([point])[0]
            crop, dst_pts, M, Minv = transform_utils.image_alignment(image,
                                                                     src_pts=src_pts,
                                                                     dst_pts=None,
                                                                     dsize=(-1, -1),
                                                                     method="lstsq")
            h, w = crop.shape[:2]
            crop_file = os.path.join(crop_root, "{}_{:0=5d}_alignment.jpg".format(label, i))
            if w > 3 * h: cv2.imwrite(crop_file, crop)
        elif crop_root:
            crop = image_utils.get_box_crop(image, box=boxes[0])
            crop_file = os.path.join(crop_root, "{}_{:0=5d}_crop.jpg".format(label, i))
            h, w = crop.shape[:2]
            if w > 3 * h: cv2.imwrite(crop_file, crop)
        json_file = os.path.join(out_json, json_name)
        build_labelme.maker_labelme(json_file, points=[point], labels=[label], image_name=image_name,
                                    image_size=(image_w, image_h))
        if vis:
            image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name=[label])
            image = image_utils.draw_key_point_in_image(image, [point], vis_id=True)
            if crop_root: image_utils.cv_show_image("crop", crop, delay=10)
            image_utils.cv_show_image("image", image, delay=delay)
    return


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/tmp/tmp/水表数字识别/Water Meters Dataset/WaterMeters/Water Meters Dataset_images_datasets"
    crop_root = "/home/PKing/nasdata/tmp/tmp/水表数字识别/Water Meters Dataset/WaterMeters/crop"
    out_json = "/home/PKing/nasdata/tmp/tmp/水表数字识别/Water Meters Dataset/WaterMeters/json"
    anno_file = "/home/PKing/nasdata/tmp/tmp/水表数字识别/Water Meters Dataset/data.csv"
    WaterMeters(image_dir, anno_file, out_json=out_json, crop_root=crop_root, vis=True)
