# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-04-25 16:20:17
    @Brief  :
"""
import copy
import cv2
from pybaseutils import image_utils, file_utils, debug


if __name__ == "__main__":
    file = "../data/test_image/grid1.png"
    dsize = (320, 320)
    image = cv2.imread(file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # dest = image_utils.resize_scale_image(image, size=max(dsize), use_length=False)
    dest = image_utils.resize_image_padding(image, size=dsize, use_length=True, color=(0, 255, 0))
    print("image：{}".format(image.shape))
    print("dest ：{}".format(dest.shape))
    image_utils.cv_show_image("image", image, delay=1)
    image_utils.cv_show_image("dest", dest)
