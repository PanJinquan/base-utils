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


class ParseDataset(object):
    def __init__(self, prb_thresh=0.5, iou_thresh=0.3):
        """
        :param file:
        :param image_dir:
        :param is_detect:
        :param image_sub:
        :param shuffle:
        """
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="mtcnn")
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="dfsd")
        self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="yolov5_person")
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

    def parse_data(self, image_dir, out_dir="", scale=[], square=False, padding=False, flag="20220200", max_num=None,
                   shuffle=True, vis=True):
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
        if max_num: nums_sample = min(max_num, nums_sample)
        for i in tqdm(range(nums_sample)):
            image_name, label = item_list[i]
            image_file = os.path.join(image_dir, image_name)
            if not os.path.exists(image_file):
                print("no path:{}".format(image_file))
                continue
            image = image_utils.read_image(image_file)
            if image is None:
                print("empty file:{}".format(image_file))
                continue
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_score, labels = self.detector.detect(rgb, vis=False)
            if len(bbox_score) == 0: continue
            dets = np.hstack((bbox_score, labels.reshape(-1, 1)))
            image_id = "{}_{:0=6d}".format(flag, i) if flag else os.path.basename(image_name).split(".")[0]
            self.save_crops(image, dets, image_id, scale=scale, square=square, padding=padding, out_dir=out_dir,
                            vis=vis)

    def faceboxes2bodyup(self, xyxy, shift=(0, 0.5), scale=(1.0, 1.0)):
        """通过人脸框获得上半身"""
        if not isinstance(xyxy, np.ndarray): xyxy = np.asarray(xyxy)
        cxcywh = image_utils.xyxy2cxcywh(xyxy)
        dcxcywh = cxcywh.copy()
        dcxcywh[:, 0] = dcxcywh[:, 0] + dcxcywh[:, 2] * shift[0]
        dcxcywh[:, 1] = dcxcywh[:, 1] + dcxcywh[:, 3] * shift[1]
        dxyxy = image_utils.cxcywh2xyxy(dcxcywh)
        dxyxy = image_utils.extend_xyxy(dxyxy, scale=scale)
        return dxyxy

    def get_boxes_up(self, xyxy, scale=(), cut=0.3):
        """获得boxes上半部分"""
        dxyxy = []
        for i in range(len(xyxy)):
            xmin, ymin, xmax, ymax = xyxy[i]
            w, h = (xmax - xmin), (ymax - ymin)
            ymax = max(ymin + h * cut, ymin + w)
            dxyxy.append([xmin, ymin, xmax, ymax])
        dxyxy = np.asarray(dxyxy)
        if scale: dxyxy = image_utils.extend_xyxy(dxyxy, scale=scale)
        return dxyxy

    def save_crops(self, image, dets, image_id, scale=[], square=False, padding=False, out_dir=None, vis=False):
        boxes = dets[:, 0:4]
        conf = dets[:, 4:5]
        labels = dets[:, 5]
        if square:
            boxes = image_utils.get_square_bboxes(boxes, use_max=True, baseline=-1)
        if scale:
            # boxes = image_utils.extend_xyxy(boxes, scale=scale)
            # boxes = self.faceboxes2bodyup(boxes, scale=scale)
            boxes = self.get_boxes_up(boxes, scale=scale)
        if padding:
            crops = image_utils.get_bboxes_crop_padding(image, boxes)
        else:
            crops = image_utils.get_bboxes_crop(image, boxes)
        if vis:
            m = image_utils.draw_image_detection_bboxes(image.copy(), boxes, conf, labels, class_name=self.names)
            image_utils.cv_show_image("image", m, use_rgb=False, delay=10)
        for i, img in enumerate(crops):
            name = self.names[int(labels[i])] if self.names else labels[i]
            if out_dir:
                # img_file = file_utils.create_dir(out_dir, name, "{}_{:0=3d}.jpg".format(image_id, i))
                img_file = file_utils.create_dir(out_dir, None, "{}_{:0=3d}.jpg".format(image_id, i))
                cv2.imwrite(img_file, img)
            if vis: image_utils.cv_show_image("crop", img, use_rgb=False, delay=0)


def demo_for_image_dir(image_dir="", out_dir=""):
    # image_dir = "/home/dm/nasdata/release/tmp/PyTorch-Classification-Trainer/data/test_image"
    # out_dir = image_dir + "-crops"
    dataset = ParseDataset()
    scale = [1.1, 1.1]
    square = False
    padding = False
    flag = file_utils.get_time("s").replace("2023", "2018")
    # flag = ""
    data_info = dataset.parse_data(image_dir=image_dir, out_dir=out_dir, scale=scale, square=square,
                                   padding=padding, flag=flag, shuffle=False, vis=False)


def demo_for_image_root():
    root = "/home/dm/nasdata/dataset/tmp/smoking-calling/train"
    out_root = "/home/dm/nasdata/dataset/tmp/smoking/smoking-person/dataset-v4/train"
    subs = file_utils.get_sub_paths(root)
    for sub in subs:
        image_dir = os.path.join(root, sub)
        out_dir = os.path.join(out_root, sub)
        demo_for_image_dir(image_dir, out_dir)


if __name__ == '__main__':
    """检测face person并裁剪"""
    # demo_for_image_root()
    demo_for_image_dir()
