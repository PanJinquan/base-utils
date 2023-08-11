# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: AlphaPose
# @Author : Pan
# @E-mail : 390737991@qq.com
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
from pybaseutils.converter import build_voc
from pybaseutils.dataloader import parser_labelme


class Labelme2VOC(object):
    """Convert Labelme to VOC dataset format"""

    def __init__(self, image_dir, anno_dir):
        """
        :param image_dir: 图片目录(*.json)
        :param anno_dir:  标注文件目录
        """
        self.image_dir = image_dir
        self.dataset = parser_labelme.LabelMeDataset(filename=None,
                                                     data_root=None,
                                                     anno_dir=anno_dir,
                                                     image_dir=image_dir,
                                                     class_name=None,
                                                     use_rgb=False,
                                                     check=False,
                                                     phase="val",
                                                     shuffle=False)

    def build_dataset(self, out_root, class_dict={}, out_image_dir=None, crop=False, rename=False, vis=True):
        """
        :param out_root: VOC输出根目录
        :param class_dict: label映射 list或dict，如果label不在class_dict中，则使用原始label
        :param out_image_dir: 保存 JPEGImages
        :param crop: 是否进行目标裁剪
        :param rename: 是否重命名
        :param vis: 是否可视化
        :return:
        """
        out_xml_dir = os.path.join(out_root, "Annotations")
        out_crop_dir = os.path.join(out_root, "crops")
        class_set = []
        for i in tqdm(range(len(self.dataset))):
            data = self.dataset.__getitem__(i)
            # data = self.dataset.__getitem__(307)
            image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
            anno_file = data["anno_file"]
            image_file = data["image_file"]
            image_shape = image.shape
            if len(labels) == 0:
                # file_utils.remove_file(anno_file)
                # file_utils.remove_file(image_file)
                print("empty dst_result:{}".format(image_file))
                continue
            format = os.path.basename(image_file).split(".")[-1]
            image_id = os.path.basename(image_file)[:-len(format) - 1]
            if rename:
                image_id = "{}_{:0=4d}".format(rename, i)
                format = "jpg"
            newname = "{}.{}".format(image_id, format)
            xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
            class_set = list(set(class_set + labels))
            objects = build_voc.create_objects(bboxes, labels, keypoints=None, class_name=class_dict)
            build_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
            if crop and out_crop_dir:
                self.save_object_crops(objects, image, out_crop_dir, image_id)
            if out_image_dir:
                dst_file = file_utils.create_dir(out_image_dir, None, newname)
                file_utils.copy_file(image_file, dst_file)
                # cv2.imwrite(dst_file, image)
            if vis:
                self.show_object_image(image, objects)
        if not out_image_dir: out_image_dir = self.image_dir
        file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                                  only_id=False, shuffle=False, max_num=None)
        class_set = sorted(class_set)
        print("have class_set:{}\n{}".format(len(class_set), class_set))

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
            image = image_utils.draw_image_bboxes_text(image, [box], [label], thickness=3, fontScale=1.2,
                                                       color=(0, 255, 0), drawType="chinese")
        image_utils.cv_show_image("image", image, use_rgb=False, delay=0)


if __name__ == "__main__":
    json_dir = "/home/dm/nasdata/dataset/tmp/fall/fall-v3/json"
    out_root = os.path.dirname(json_dir)
    image_dir = os.path.join(out_root, "JPEGImages")
    class_dict = {}
    # class_dict = {"person": "up", "squat": "bending", "fall": "down"}
    lm = Labelme2VOC(image_dir, json_dir)
    lm.build_dataset(out_root, class_dict=class_dict, vis=False, crop=False)
