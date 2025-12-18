# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.utils.matching import iou_distance


class YOLOv8():
    def __init__(self, model_file, task="segment"):
        self.model = YOLO(model_file, task=task)

    def predict(self, image_file, imgsz=640, save=True):
        results = self.model.predict(image_file, imgsz=imgsz, save=save)  # predict on an image
        return results

if __name__ == "__main__":
    from pybaseutils import image_utils

    image_utils.get_contours_iou() # 计算轮廓的IOU
    image_utils.shrink_polygon_pyclipper() # 缩放多边形
