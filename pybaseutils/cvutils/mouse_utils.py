# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from typing import Callable
from pybaseutils import image_utils


class DrawImageMouse(object):
    """使用鼠标绘图"""

    def __init__(self, max_point=-1, line_color=(0, 0, 255), text_color=(255, 0, 0), thickness=2):
        """
        :param max_point: 最多绘图的点数，超过后将绘制无效；默认-1表示无限制
        :param line_color: 线条的颜色
        :param text_color: 文本的颜色
        :param thickness: 线条粗细
        """
        self.max_point = max_point
        self.line_color = line_color
        self.text_color = text_color
        self.text_size = max(int(thickness * 0.4), 0.5)
        self.focus_color = (0, 255, 0)  # 鼠标焦点的颜色
        self.focus_size = max(int(thickness * 2), 6)  # 鼠标焦点的颜色
        self.thickness = thickness
        self.key = -1  # 键盘值
        self.orig = None  # 原始图像
        self.last = None  # 上一帧
        self.next = None  # 下一帧或当前帧
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)  # 鼠标绘制点集合

    def clear(self):
        self.key = -1
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)
        if self.orig is not None: self.last = self.orig.copy()
        if self.orig is not None: self.next = self.orig.copy()

    def get_polygons(self):
        """获得多边形数据"""
        return self.polygons

    def task(self, image, callback: Callable, winname="winname"):
        """
        鼠标监听任务
        :param image: 图像
        :param callback: 鼠标回调函数
        :param winname: 窗口名称
        :return:
        """
        self.orig = image.copy()
        self.last = image.copy()
        self.next = image.copy()
        cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(winname, callback, param={"winname": winname})
        while True:
            self.key = self.show_image(winname, self.next, delay=25)
            print("key={}".format(self.key))
            if (self.key == 13 or self.key == 32) and len(self.polygons) > 0:  # 按空格32和回车键13表示完成绘制
                break
            elif self.key == 27:  # 按ESC退出程序
                exit(0)
            elif self.key == 99:  # 按键盘c重新绘制
                self.clear()
        # cv2.destroyAllWindows()
        cv2.setMouseCallback(winname, self.event_default)

    def event_default(self, event, x, y, flags, param):
        pass

    def event_draw_rectangle(self, event, x, y, flags, param):
        """绘制矩形框"""
        if len(self.polygons) == 0: self.polygons = np.zeros(shape=(2, 2), dtype=np.int32)  # 多边形轮廓
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            self.next = self.last.copy()
            self.polygons[0, :] = point
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            print("2-EVENT_FLAG_LBUTTON")
            self.next = self.last.copy()
            cv2.circle(self.next, self.polygons[0, :], radius=self.focus_size, color=self.focus_color,
                       thickness=self.thickness)
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            print("3-EVENT_LBUTTONUP")
            self.next = self.last.copy()
            self.polygons[1, :] = point
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)
        print("location:{},have:{}".format(point, len(self.polygons)))

    def event_draw_polygon(self, event, x, y, flags, param):
        """绘制多边形"""
        exceed = self.max_point > 0 and len(self.polygons) >= self.max_point
        self.next = self.last.copy()
        point = (x, y)
        text = str(len(self.polygons))
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            cv2.putText(self.next, text, point, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color,
                        self.thickness)
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)
            if not exceed:
                self.last = self.next
                self.polygons = np.concatenate([self.polygons, np.array(point).reshape(1, 2)])
        else:
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)
        print("location:{},have:{}".format(point, len(self.polygons)))

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

    def draw_image_rectangle_on_mouse(self, image, winname="draw_rectangle"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: box is[xmin,ymin,xmax,ymax]
        """
        self.task(image, callback=self.event_draw_rectangle, winname=winname)
        polygons = self.get_polygons()
        box = self.polygons2box(polygons)
        return box

    def draw_image_polygon_on_mouse(self, image, winname="draw_polygon"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: polygons is (N,2)
        """
        self.task(image, callback=self.event_draw_polygon, winname=winname)
        polygons = self.get_polygons()
        return polygons


def draw_image_rectangle_on_mouse_example(image_file, winname="draw_rectangle"):
    """
    获得鼠标绘制的矩形框
    :param image_file:
    :param winname: 窗口名称
    :return: box=[xmin,ymin,xmax,ymax]
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制矩形框rect
    mouse = DrawImageMouse()
    box = mouse.draw_image_rectangle_on_mouse(image, winname=winname)
    # 裁剪矩形区域,并绘制最终的矩形框
    roi: np.ndarray = image[box[1]:box[3], box[0]:box[2]]
    if roi.size > 0: mouse.show_image("Image ROI", roi)
    image = image_utils.draw_image_boxes(image, [box], color=(0, 0, 255), thickness=2)
    mouse.show_image(winname, image, delay=0)
    return box


def draw_image_polygon_on_mouse_example(image_file, winname="draw_polygon"):
    """
    获得鼠标绘制的多边形
    :param image_file:
    :param winname: 窗口名称
    :return: polygons is (N,2)
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制多边形
    mouse = DrawImageMouse(max_point=-1)
    polygons = mouse.draw_image_polygon_on_mouse(image, winname=winname)
    image = image_utils.draw_image_points_lines(image, polygons, thickness=2)
    mouse.show_image(winname, image, delay=0)
    return polygons


if __name__ == '__main__':
    image_file = "../../data/test.png"
    # 绘制矩形框
    # out = draw_image_rectangle_on_mouse_example(image_file)
    # 绘制多边形
    out = draw_image_polygon_on_mouse_example(image_file)
    print(out)
