# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from ultralytics import YOLO


class YOLOv8():
    def __init__(self, model_file, task="segment"):
        self.model = YOLO(model_file, task=task)

    def predict(self, image_file, imgsz=640, save=True):
        results = self.model.predict(image_file, imgsz=imgsz, save=save)  # predict on an image
        return results


if __name__ == "__main__":
    model_file = "weights/best.pt"
    image_file = "images/image1.jpg"
    infer = YOLOv8(model_file)
    results = infer.predict(image_file)
