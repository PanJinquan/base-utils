import os

import cv2
from PIL import Image, ImageFont
from pybaseutils import image_utils, file_utils

if __name__ == "__main__":
    """streamlit run web.py"""
    image_file = "/home/PKing/nasdata/release/tmp/Pytorch-Character-Recognition/data/test_image/test02.jpg"
    image = cv2.imread(image_file)
    boxes = [[100, 50, 500, 100]]
    names = ["你好ASDV"]
    image = image_utils.draw_image_boxes_texts(image, boxes, names, color=(0, 255, 0), drawType="ch",
                                               thickness=2, fontScale=1.0, top=False)
    # cv2.imwrite("style.png", image)
    image_utils.cv_show_image("image", image, delay=0)
