# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, "/home/dm/nasdata/release/detector/object-detector")
import random
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, pandas_utils
from libs.detector.detector import Detector

class_name = {0: "neutral",
              1: "happy",
              2: "excited",
              3: "surprised",
              4: "sad",
              5: "disgust",
              6: "fear",
              7: "confused",
              8: "tired",
              9: "angry",
              10: "serious"
              }


class ParseDataset(object):
    def __init__(self, prb_thresh=0.9, iou_thresh=0.3):
        """
        :param file:
        :param image_dir:
        :param is_detect:
        :param image_sub:
        :param shuffle:
        """
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="mtcnn")
        self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="dfsd")

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

    def parse_data(self, image_dir, out_dir="", flag="20220200", square=True,
                   extend=[2.0, 2.0], max_num=None, shuffle=True, vis=True):
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
        if max_num:
            nums_sample = min(max_num, nums_sample)
        dst_item_list = []
        for i in tqdm(range(nums_sample)):
            image_file, label = item_list[i]
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                print("no path:{}".format(image_path))
                continue
            image = image_utils.read_image(image_path)
            if image is None:
                print("empty file:{}".format(image_path))
                continue
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets, _ = self.detector.detect(rgb, vis=False)
            if len(dets) == 0: continue
            boxes = dets[:, 0:4]
            if square: boxes = image_utils.get_square_bboxes(boxes, use_max=True)
            if extend: boxes = image_utils.extend_xyxy(boxes, scale=extend)
            xmin, ymin, xmax, ymax = np.asarray(boxes, dtype=np.int32)[0]
            sub, name = image_file.split("/")
            name = "{}_{:0=6d}.jpg".format(flag, i)
            # item = [os.path.join(label, name), xmin, ymin, xmax, ymax]
            item = [image_file, label, xmin, ymin, xmax, ymax]
            box = [xmin, ymin, xmax, ymax]
            dst_item_list.append(item)
            if out_dir:
                # orig_file = file_utils.create_dir(os.path.join(out_dir, "origin"), label, name)
                # file_utils.copy_file(image_path, orig_file)
                crop_file = file_utils.create_dir(os.path.join(out_dir, "crops"), label, name)
                crop = image_utils.get_bbox_crop(image, box)
                cv2.imwrite(crop_file, crop)
            if vis:
                image = image_utils.draw_image_boxes(image, [box])
                image = image_utils.draw_texts(image, [[xmin + 5, ymin + 10]], texts=[label])
                image_utils.cv_show_image("image", image)
        if not out_dir: out_dir = image_dir
        filename = file_utils.create_dir(out_dir, None, "label.txt")
        file_utils.write_data(filename, dst_item_list, split=",")
        return dst_item_list


def main():
    image_dir = "/home/dm/nasdata/dataset/csdn/smoking/smokingVSnotsmoking/training_data"
    out_dir = "/home/dm/nasdata/dataset/csdn/smoking/dataset1"
    dataset = ParseDataset()
    data_info = dataset.parse_data(image_dir=image_dir, out_dir=out_dir, flag="202200000", shuffle=False, vis=False)


if __name__ == '__main__':
    main()
