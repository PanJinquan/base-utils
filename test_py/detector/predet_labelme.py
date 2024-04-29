# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, "/home/PKing/nasdata/release/detector/object-detector")
import random
import cv2
import numpy as np
import traceback
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, pandas_utils
from pybaseutils.converter import build_labelme
from libs.detector.detector import Detector


class ParseDataset(object):
    def __init__(self, prb_thresh=0.5, iou_thresh=0.3):
        """
        :param file:
        :param image_dir:
        :param is_detect:
        :param image_sub:
        :param shuffle:
        """
        self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="mtcnn")
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="dfsd")
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="yolov5_person")
        self.names = self.detector.class_names

    def get_item_list_class(self, image_dir, shuffle=True):
        """
        数据格式：
        image_dir
           ├── A
           │ ├── 0001.jpg
           │ └── 0002.jpg
           └── B
             ├── 0001.jpg
             └── 0002.jpg
        :param image_dir:
        :param shuffle:
        :return:
        """
        images = file_utils.get_images_list(image_dir)
        images = file_utils.get_sub_list(images, image_dir)
        item_list = [[path, os.path.dirname(path)] for path in images]
        if shuffle:
            random.seed(200)
            random.shuffle(item_list)
        print("have dataset:{}".format(len(item_list)))
        return item_list

    def map_class_names(self, image_file, bbox_score, labels, max_nums=-1):
        # if max_nums > 0:
        #     bbox_score = bbox_score[0:min(len(bbox_score), max_nums)]
        #     labels = labels[0:min(len(bbox_score), max_nums)]
        # TODO 直接映射label
        # names = [self.names[int(l)] for l in labels]
        # TODO 子文件夹是label
        sub = os.path.basename(os.path.dirname(image_file))
        # class_dict = {"no_yawn": "undrowse", "yawn": "drowse"}
        # class_dict = {"undrowse": "undrowse", "drowse": "drowse"}
        # names = [class_dict[sub]] * len(bbox_score)
        names = [sub] + ["undrowsy"] * (len(bbox_score) - 1)
        # TODO others
        return names, bbox_score

    def parse_data(self, image_dir, out_dir="", flag="20220200", shuffle=False, vis=True):
        """
        数据格式：
        image_dir
           ├── A
           │ ├── 0001.jpg
           │ └── 0002.jpg
           └── B
             ├── 0001.jpg
             └── 0002.jpg
        :param image_dir:
        :param out_dir:
        :param flag:
        :param max_num:
        :param shuffle:
        :param vis:
        :return:
        """
        item_list = self.get_item_list_class(image_dir=image_dir, shuffle=shuffle)
        nums_sample = len(item_list)
        for i in tqdm(range(nums_sample)):
            # i = 6
            try:
                image_name, label = item_list[i]
                image_file = os.path.join(image_dir, image_name)
                if not os.path.exists(image_file):
                    print("no path:{}".format(image_file))
                    continue
                image = image_utils.read_image(image_file)
                if image is None:
                    print("empty file:{}".format(image_file))
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bbox_score, labels = self.detector.detect(rgb, vis=False)
                if len(bbox_score) == 0: continue
                names, bbox_score = self.map_class_names(image_file, bbox_score, labels, max_nums=1)
                image_id = "{}_{:0=6d}".format(flag, i) if flag else os.path.basename(image_name).split(".")[0]
                # image_ids = "{}_{:0=6d}".format(flag,int(os.path.basename(image_name).split(".")[0]))
                self.save_lableme(image, bbox_score, names, image_file, image_id, out_dir=out_dir, vis=vis)
            except:
                print("i={},file={}".format(i, image_file))
                traceback.print_exc()
                exit(0)

    def save_lableme(self, image, bbox_score, names, image_file, image_id, out_dir, vis=False):
        bbox_score = np.asarray(bbox_score)
        boxes = bbox_score[:, 0:4]
        conf = bbox_score[:, 4:5]
        if out_dir:
            H, W = image.shape[:2]
            image_size = (W, H)
            points = image_utils.boxes2polygons(boxes)
            json_file = file_utils.create_dir(out_dir, None, "{}.json".format(image_id))
            img_file = file_utils.create_dir(out_dir, None, "{}.jpg".format(image_id))
            file_utils.remove_file(img_file)
            build_labelme.maker_labelme(json_file, points, names, img_file, image_size)
            # cv2.imwrite(img_file, image)
            file_utils.copy_file(image_file, img_file)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name=names, thickness=1, fontScale=0.8)
            image_utils.cv_show_image("image", image)


def demo_for_image_dir(image_dir="", out_dir="", flag=""):
    dataset = ParseDataset()
    name = file_utils.get_time("s").replace("2023", "2018")
    if flag: name = "{}_{}".format(flag, name)
    data_info = dataset.parse_data(image_dir=image_dir, out_dir=out_dir, flag=name, shuffle=False, vis=False)


def demo_for_image_root():
    root = "/home/PKing/nasdata/dataset/tmp/Drowsy-Driving/dataset/Drowsy-Driving-Det1/src1"
    out_root = "/home/PKing/nasdata/dataset/tmp/Drowsy-Driving/dataset/Drowsy-Driving-Det1/train"
    subs = file_utils.get_sub_paths(root)
    for i, sub in enumerate(subs):
        image_dir = os.path.join(root, sub)
        # out_dir = os.path.join(out_root, sub)
        out_dir = out_root
        demo_for_image_dir(image_dir, out_dir, flag="")


if __name__ == '__main__':
    """检测face person并裁剪"""
    demo_for_image_root()
    # demo_for_image_dir()
