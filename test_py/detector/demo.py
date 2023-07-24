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
from pybaseutils.maker import maker_labelme
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

    def parse_data(self, image_dir, shuffle=False, vis=True):
        image_list = file_utils.get_files_lists(image_dir, shuffle=shuffle)
        for image_file in image_list:
            image = image_utils.read_image(image_file)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_score, labels = self.detector.detect(rgb, vis=False)
            bbox_score = np.asarray(bbox_score)
            boxes = bbox_score[:, 0:4]
            conf = bbox_score[:, 4:5]
            if vis:
                image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name=labels, thickness=2, fontScale=0.8)
                image_utils.cv_show_image("image", image)


if __name__ == '__main__':
    """检测face person并裁剪"""
    image_dir = "/home/PKing/nasdata/dataset/tmp/drowsy-driving/drowsy-driving/DDDataset-Det1/JPEGImages"
    dataset = ParseDataset()
    dataset.parse_data(image_dir)
