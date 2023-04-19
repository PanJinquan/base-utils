# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import image_utils
from pybaseutils.cvutils import mouse_utils


def draw_image_rectangle_on_mouse_example(image_file, winname="image"):
    """
    获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
    :param image_file:
    :return:
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制矩形框rect
    mouse = mouse_utils.DrawImageMouse(winname=winname)
    box = mouse.draw_image_rectangle_on_mouse(image)
    # 裁剪矩形区域,并绘制最终的矩形框
    roi: np.ndarray = image[box[1]:box[3], box[0]:box[2]]
    if roi.size > 0: mouse.show_image("Image ROI", roi)
    image = image_utils.draw_image_boxes(image, [box], color=(0, 0, 255), thickness=2)
    mouse.show_image(winname, image, delay=0)
    return box


def draw_image_polygon_on_mouse_example(image_file, winname="image"):
    """
    获得鼠标绘制的多边形box=[xmin,ymin,xmax,ymax]
    :param image_file:
    :return:
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制多边形
    mouse = mouse_utils.DrawImageMouse(winname=winname, max_point=4)
    polygons = mouse.draw_image_polygon_on_mouse(image)
    image = image_utils.draw_image_points_lines(image, polygons, thickness=2)
    mouse.show_image(winname, image, delay=0)
    return polygons


if __name__ == '__main__':
    image_path = "/media/dm/新加卷/SDK/base-utils/data/test.png"
    # out = draw_image_rectangle_on_mouse_example(image_path)
    out = draw_image_polygon_on_mouse_example(image_path)
    print(out)
