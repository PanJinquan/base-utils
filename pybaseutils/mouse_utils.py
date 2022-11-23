# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from typing import Callable
from pybaseutils import image_utils


class DrawImageMouse(object):
    """使用鼠标绘图"""

    def __init__(self, winname, max_point=-1, color=(0, 0, 255), thickness=2):
        """
        :param winname: 窗口名称
        :param max_point: 最多绘图的点数，超过后将绘制无效；默认-1表示无限制
        :param color: 框的颜色
        :param thickness: 框的线条粗细
        """
        self.winname = winname
        self.max_point = max_point
        self.image = None
        self.polygons = None
        self.color = color
        self.thickness = thickness
        self.key = -1

    def task(self, image, callback: Callable, destroy=False):
        self.image = image
        cv2.namedWindow(self.winname, flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.winname, callback)
        while True:
            key = self.show_image(self.winname, self.image, delay=20)
            print("key={}".format(key))
            if key == 13 or key == 32:  # 按空格和回车键退出
                break
        if destroy: cv2.destroyAllWindows()
        cv2.setMouseCallback(self.winname, self.event_default)

    def event_default(self, event, x, y, flags, param):
        pass

    def event_draw_rectangle(self, event, x, y, flags, param):
        """绘制矩形框"""
        if self.polygons is None: self.polygons = np.zeros(shape=(2, 2), dtype=np.int32)  # 多边形轮廓
        cur_image = self.image.copy()
        cur_point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            self.polygons[0, :] = cur_point
            cv2.circle(cur_image, cur_point, radius=5, color=(0, 255, 0), thickness=self.thickness)
            self.key = self.show_image(self.winname, cur_image, delay=5)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            print("2-EVENT_FLAG_LBUTTON")
            cv2.circle(cur_image, self.polygons[0, :], radius=4, color=(0, 255, 0), thickness=self.thickness)
            cv2.circle(cur_image, cur_point, radius=4, color=(0, 255, 0), thickness=self.thickness)
            cv2.rectangle(cur_image, self.polygons[0, :], cur_point, color=self.color, thickness=self.thickness)
            self.key = self.show_image(self.winname, cur_image, delay=-1)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            print("3-EVENT_LBUTTONUP")
            self.polygons[1, :] = cur_point
            cv2.rectangle(cur_image, self.polygons[0, :], cur_point, color=self.color, thickness=self.thickness)
            self.key = self.show_image(self.winname, cur_image, delay=-1)
            if np.sum(self.polygons[0, :] - self.polygons[1, :]) != 0:
                xmin, ymin, xmax, ymax = self.polygons2box(self.polygons)
                cut = self.image[ymin:ymax, xmin:xmax]
                self.show_image('ROI', cut, delay=-1)
        print("point:{},have:{}".format(cur_point, len(self.polygons)))

    def event_draw_polygon(self, event, x, y, flags, param):
        """绘制多边形"""
        if self.polygons is None: self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)  # 多边形轮廓
        exceed = self.max_point > 0 and len(self.polygons) >= self.max_point
        cur_image = self.image.copy()
        cur_point = (x, y)
        text = str(len(self.polygons))
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            cv2.circle(cur_image, cur_point, radius=5, color=(0, 255, 0), thickness=self.thickness)
            cv2.putText(cur_image, text, cur_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if len(self.polygons) > 0:
                cv2.line(cur_image, self.polygons[-1, :], cur_point, color=(0, 255, 0), thickness=self.thickness)
            self.key = self.show_image(self.winname, cur_image, delay=10)
            if not exceed: self.image = cur_image
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            print("2-EVENT_FLAG_LBUTTON")
            if not exceed: self.polygons = np.concatenate([self.polygons, np.array(cur_point).reshape(1, 2)])
        else:
            cv2.circle(cur_image, cur_point, radius=5, color=(0, 255, 0), thickness=self.thickness)
            if len(self.polygons) > 0:
                cv2.line(cur_image, self.polygons[-1, :], cur_point, color=(0, 255, 0), thickness=self.thickness)
            self.key = self.show_image(self.winname, cur_image, delay=-1)
        print("point:{},have:{}".format(cur_point, len(self.polygons)))

    def get_polygons(self):
        """获得多边形数据"""
        return self.polygons

    @staticmethod
    def polygons2box(polygons):
        """将多边形转换为矩形框"""
        xmin = min(polygons[:, 0])
        ymin = min(polygons[:, 1])
        xmax = max(polygons[:, 0])
        ymax = max(polygons[:, 1])
        return [xmin, ymin, xmax, ymax]

    def show_image(self, title, image, delay=5):
        """显示图像"""
        cv2.imshow(title, image)
        key = cv2.waitKey(delay=delay) if delay >= 0 else -1
        return key

    def draw_image_rectangle_on_mouse(self, image, destroy=False):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :return: box is[xmin,ymin,xmax,ymax]
        """
        self.task(image, callback=self.event_draw_rectangle, destroy=destroy)
        polygons = self.get_polygons()
        box = self.polygons2box(polygons)
        return box

    def draw_image_polygon_on_mouse(self, image, destroy=False):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :return: polygons is (N,2)
        """
        self.task(image, callback=self.event_draw_polygon, destroy=destroy)
        polygons = self.get_polygons()
        return polygons


def draw_image_rectangle_on_mouse_example(image_file, winname="image"):
    """
    获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
    :param image_file:
    :return:
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制矩形框rect
    mouse = DrawImageMouse(winname=winname)
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
    mouse = DrawImageMouse(winname=winname, max_point=4)
    polygons = mouse.draw_image_polygon_on_mouse(image)
    image = image_utils.draw_image_points_lines(image, polygons, thickness=2)
    mouse.show_image(winname, image, delay=0)
    return polygons


if __name__ == '__main__':
    image_path = "/media/dm/新加卷/SDK/base-utils/data/test.png"
    # rect = draw_image_rectangle_on_mouse_example(image_path)
    rect = draw_image_polygon_on_mouse_example(image_path)
    # print(rect)
