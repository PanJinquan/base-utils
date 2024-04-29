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
sys.path.insert(0, "/home/PKing/nasdata/release/detector/object-detector")
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.converter import build_voc
from pybaseutils.dataloader import parser_labelme
from libs.detector.detector import Detector
from pybaseutils.converter import convert_labelme2voc


class Labelme2VOC(convert_labelme2voc.Labelme2VOC):
    """Convert Labelme to VOC dataset format"""

    def __init__(self, image_dir, anno_dir, class_name=None, shuffle=False, min_points=1):
        """
        :param image_dir: 图片目录(*.json)
        :param anno_dir:  标注文件目录
        :param min_points: 当标注的轮廓点的个数小于min_points，会被剔除；负数不剔除
        """
        super(Labelme2VOC, self).__init__(image_dir, anno_dir, class_name=class_name,
                                          shuffle=shuffle, min_points=min_points)
        self.detector = Detector(prb_thresh=0.5, iou_thresh=0.3, detect_type="yolov5_hand")

    def get_hand_pen_object(self, data, vis=False):
        image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
        if len(points) == 0 or len(bboxes) == 0 or len(labels) == 0: return data
        h, w = image.shape[:2]
        have_pen = [n for n in labels if str(n).startswith("pen")]
        have_fig = [n for n in labels if str(n).startswith("finger")]
        points = [p[0] for p in points if len(p) > 0]
        points = np.asarray(points).reshape(-1, 2)  # 使用所有点计算boxes
        boxes = image_utils.polygons2boxes([points])
        if len(have_pen) == 0 and len(have_fig) > 0:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_score, labels = self.detector.detect(rgb, vis=False)
            boxes = bbox_score[0:1, 0:4] if len(bbox_score) > 0 else []
            labels = ["hand"] * len(boxes)
        elif ("pen0" in have_pen) and ("pen1" in have_pen) and len(have_fig) > 0:
            labels = ["hand_pen"]
            boxes = image_utils.extend_xyxy_similar_square(boxes, weight=0.6)
            boxes = image_utils.extend_xyxy(boxes, scale=[1.1, 1.1], valid_range=(0, 0, w, h), fixed=True, use_max=True)
        else:
            points, bboxes, labels = [], [], []
        data["point"], data["box"], data["label"] = points, boxes, labels
        if vis:
            image = image_utils.draw_image_bboxes_text(image, data["box"], boxes_name=data["label"])
            image_utils.cv_show_image("image", image)
        return data

    def get_hand_pen_tip_object(self, data, vis=False):
        image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
        if len(points) == 0 or len(bboxes) == 0 or len(labels) == 0: return data
        h, w = image.shape[:2]
        targets = {"pen_tip": [], "finger_tip": []}
        for label, point in zip(labels, points):
            if label == "pen0":
                targets["pen_tip"] = point[0]
            if label == "finger2":
                targets["finger_tip"] = point[0]
        points = [p[0] for p in points if len(p) > 0]
        points = np.asarray(points).reshape(-1, 2)  # 使用所有点计算boxes
        hand = image_utils.polygons2boxes([points])[0]
        # (1)使用手部box计算指尖笔尖框的大小
        # baseline = max(hand[2] - hand[0], hand[3] - hand[1])
        # baseline = max(baseline * 0.3, 80)
        # if abs(hand[3] - hand[1]) > 2 and abs(hand[3] - hand[1]) > 2:
        #     rates = (hand[2] - hand[0]) / (hand[3] - hand[1])  # w/h
        #     radial = [baseline * rates / 2, baseline / 2]  # (x,y)
        # else:
        #     radial = [baseline / 2, baseline / 2]
        # (1)使用图像大小，计算指尖笔尖框的大小
        baseline = max(h, w) * 0.1
        if abs(hand[3] - hand[1]) > 2 and abs(hand[3] - hand[1]) > 2:
            rates = (hand[2] - hand[0]) / (hand[3] - hand[1])  # w/h
            rates = min(rates, 1.3)
            rates = max(rates, 1 / 1.3)
            radial = [baseline * rates / 2, baseline / 2]  # (x,y)
            print(rates)
        else:
            radial = [baseline / 2, baseline / 2]

        labels, center = [], []
        if len(targets['pen_tip']) > 0:  # 含有笔尖数据
            point = targets['pen_tip']
            labels += ["pen_tip"]
            center += [[point[0], point[1], radial[0], radial[1]]]
        elif len(targets['pen_tip']) == 0 and len(targets['finger_tip']) > 0:  # 没有笔尖，只有指尖
            point = targets['finger_tip']
            labels += ["finger_tip"]
            center += [[point[0], point[1], radial[0], radial[1]]]
        if len(center) > 0:
            boxes = image_utils.cxcywh2xyxy(center)
            boxes = image_utils.extend_xyxy_similar_square(boxes, weight=0.6)
            boxes = image_utils.extend_xyxy(boxes, scale=[1.2, 1.2], valid_range=(0, 0, w, h), fixed=True, use_max=True)
        else:
            points, boxes, labels = [], [], []
        data["point"], data["box"], data["label"] = points, boxes, labels
        if vis:
            image = image_utils.draw_image_bboxes_text(image, data["box"], boxes_name=data["label"])
            image_utils.cv_show_image("image", image)
        return data

    def get_object_detection(self, data):
        """
        data = {"image": image, "point": point, "box": box, "label": label, "groups": groups,
                "image_file": image_file, "anno_file": anno_file, "width": width, "height": height}
        :param data:
        :return:
        """
        # data = self.get_hand_pen_object(data)
        data = self.get_hand_pen_tip_object(data)
        return data


if __name__ == "__main__":
    # json_dir = "/home/PKing/nasdata/dataset-dmai/handwriting/word-det/word-v8/json"
    # json_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-test1-2/json"
    json_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-v1-v9/json"
    data_root = os.path.dirname(json_dir)
    # image_dir = os.path.join(data_root, "JPEGImages")
    image_dir = os.path.join(data_root, "images")
    out_root = os.path.join(data_root, "VOC")
    class_dict = None
    # class_dict = {"person": "up", "squat": "bending", "fall": "down"}
    lm = Labelme2VOC(image_dir, json_dir, min_points=-1)
    lm.build_dataset(out_root, class_dict=class_dict, out_img=True, vis=False, crop=False)
