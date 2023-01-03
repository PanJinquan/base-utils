# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import cv2
from pybaseutils import image_utils, file_utils

if __name__ == "__main__":
    image_file = "test.png"
    image = image_utils.read_image(image_file)
    boxes = [[10, 50, 200, 200]]
    boxes_name = ["ABCDabcd"]
    image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name, drawType="chinese", thickness=3, fontScale=1.0)
    image = image_utils.draw_image_bboxes_text(image, boxes, boxes_name, drawType="simple", thickness=3, fontScale=1.0)
    image_utils.cv_show_image("image", image)
