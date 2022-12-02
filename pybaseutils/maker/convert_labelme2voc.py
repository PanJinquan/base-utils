# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: AlphaPose
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-14 09:15:52
# --------------------------------------------------------
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.maker import maker_voc
from pybaseutils.dataloader import parser_labelme


class LabelMeDemo(object):
    def __init__(self, json_dir, image_dir):
        """
        :param json_dir:
        :param image_dir:
        """
        self.dataset = parser_labelme.LabelMeDataset(filename=None,
                                                     data_root=None,
                                                     anno_dir=json_dir,
                                                     image_dir=image_dir,
                                                     class_name=None,
                                                     use_rgb=False,
                                                     check=False,
                                                     phase="val",
                                                     shuffle=False)

    def convert_dataset2voc(self, out_root, class_name=None, out_image_dir=None, crop=True, rename=False, vis=True):
        """
        :param out_root: VOC输出根目录
        :param class_name: label映射 list或dict
        :param out_image_dir: 保存 JPEGImages
        :param crop: 是否进行目标裁剪
        :param rename: 是否重命名
        :param vis: 是否可视化
        :return:
        """
        out_xml_dir = os.path.join(out_root, "Annotations")
        out_crop_dir = os.path.join(out_root, "crops")
        for i in tqdm(range(len(self.dataset))):
            data = self.dataset.__getitem__(i)
            image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
            image_file = data["image_file"]
            image_shape = image.shape
            if len(labels) == 0:
                print("empty dst_result:{}".format(image_file))
                continue
            format = os.path.basename(image_file).split(".")[-1]
            image_id = os.path.basename(image_file)[:-len(format) - 1]
            if rename:
                image_id = "{}_{:0=4d}".format(rename, i)
                format = "jpg"
            newname = "{}.{}".format(image_id, format)
            xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
            objects = maker_voc.create_objects(bboxes, labels, keypoints=None, class_name=class_name)
            maker_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
            if crop and out_crop_dir:
                self.save_object_crops(objects, image, out_crop_dir, image_id)
            if out_image_dir:
                dst_file = file_utils.create_dir(out_image_dir, None, newname)
                file_utils.copy_file(image_file, dst_file)
                # cv2.imwrite(dst_file, image)
            if vis:
                self.show_object_image(image, objects)
        file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                                  only_id=False, shuffle=False, max_num=None)

    def save_object_crops(self, objects, image, out_dir, image_id):
        for i, item in enumerate(objects):
            label = item["name"]
            box = item["bndbox"]
            img = image_utils.get_bbox_crop(image, bbox=box)
            file = os.path.join(out_dir, str(label), "{}_{}_{:0=3d}.jpg".format(image_id, label, i))
            file_utils.create_file_path(file)
            cv2.imwrite(file, img)

    def show_object_image(self, image, objects):
        for item in objects:
            label = item["name"]
            box = item["bndbox"]
            image = image_utils.draw_image_bboxes_text(image, [box], [label])
        image_utils.cv_show_image("image", image, use_rgb=False, delay=0)


if __name__ == "__main__":
    json_dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/zip/json"
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/zip/JPEGImages"
    out_root = os.path.dirname(json_dir)
    class_name = None
    lm = LabelMeDemo(json_dir, image_dir)
    lm.convert_dataset2voc(out_root, class_name=class_name, vis=False)
