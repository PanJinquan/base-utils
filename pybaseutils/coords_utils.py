# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-09-08 16:58:52
"""

import math
import random
import cv2
import numbers
import numpy as np


def clip_xyxy(xyxy: np.ndarray, valid_range):
    """
    限制xyxy的有效范围，避免越界
    :param xyxy:shape=(N,4),(x,y,x,y)
    :param valid_range:有效范围(xmin,ymin,xmax,ymax)
    :return:
    """
    if len(xyxy) == 0: return xyxy
    xmin, ymin, xmax, ymax = valid_range
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], xmin, xmax)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], ymin, ymax)
    return xyxy


def clip_cxcywh_minmax(cxcywh, wh_thresh, use_max=True):
    """
    限制cxcywh的(w,h)最大最小值
    :param cxcywh:shape=(N,4),(cx,cy,w,h)
    :param wh_thresh: 有效长宽的阈值(w,h)
    :param use_max: True：最大值限制, False: 最小值限制
    :return:
    """
    if len(cxcywh) == 0: return cxcywh
    if isinstance(wh_thresh, numbers.Number): wh_thresh = [wh_thresh, wh_thresh]
    if not isinstance(cxcywh, np.ndarray): cxcywh = np.asarray(cxcywh)
    centers = cxcywh.copy()
    if use_max:
        w = centers[:, 2] > wh_thresh[0]
        centers[w, 2] = wh_thresh[0]
        h = centers[:, 3] > wh_thresh[1]
        centers[h, 3] = wh_thresh[1]
    else:
        w = centers[:, 2] < wh_thresh[0]
        centers[w, 2] = wh_thresh[0]
        h = centers[:, 3] < wh_thresh[1]
        centers[h, 3] = wh_thresh[1]
    return centers


def xyxy2xywh(xyxy: np.ndarray):
    """(xmin,ymin,xmax,ymax)==>(xmin,ymin,w,h)"""
    if len(xyxy) == 0: return xyxy
    if not isinstance(xyxy, np.ndarray): xyxy = np.asarray(xyxy)
    xywh = xyxy.copy()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]  # w=xmax-xmin
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]  # w=ymax-ymin
    return xywh


def xywh2xyxy(xywh: np.ndarray):
    """(xmin,ymin,w,h)==>(xmin,ymin,xmax,ymax)"""
    if len(xywh) == 0: return xywh
    if not isinstance(xywh, np.ndarray): xywh = np.asarray(xywh)
    xyxy = xywh.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]  # xmax=xmin+w
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]  # ymax=ymin+h
    return xyxy


def xyxy2cxcywh(xyxy: np.ndarray, width=None, height=None, normalized=False):
    """(xmin, ymin, xmax, ymax)==>(cx,cy,w,h)"""
    if len(xyxy) == 0: return xyxy
    if not isinstance(xyxy, np.ndarray): xyxy = np.asarray(xyxy)
    cxcywh = xyxy.copy()
    cxcywh[:, 0] = (xyxy[:, 2] + xyxy[:, 0]) / 2  # cx
    cxcywh[:, 1] = (xyxy[:, 3] + xyxy[:, 1]) / 2  # cy
    cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0])  # w
    cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1])  # h
    if normalized:
        cxcywh[:, 0:4] = cxcywh[:, 0:4] / (width, height, width, height)
    return cxcywh


def cxcywh2xyxy(cxcywh: np.ndarray, width=None, height=None, normalized=False):
    """(cx,cy,w,h)==>xmin, ymin, xmax, ymax)"""
    if len(cxcywh) == 0: return cxcywh
    if not isinstance(cxcywh, np.ndarray): cxcywh = np.asarray(cxcywh)
    xyxy = cxcywh.copy()
    xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2  # top left x
    xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2  # top left y
    xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2  # bottom right y
    if normalized:
        xyxy[:, 0:4] = xyxy[:, 0:4] * (width, height, width, height)
    return xyxy


def extend_xyxy(xyxy: np.ndarray, scale=[1.0, 1.0], valid_range=[], fixed=False, use_max=True):
    """
    :param bboxes: [[xmin, ymin, xmax, ymax]]
    :param scale: [sx,sy]==>(W,H)
    :param valid_range:有效范围(xmin,ymin,xmax,ymax)
    :param fixed: 长宽是否按照相同大小扩展
                 当长宽比相差比较大，直接scale会导致短边扩展不明显，
                 使用fixed的长宽会按照相同大小扩展
    :return:
    """
    if len(xyxy) == 0 or len(scale) == 0: return xyxy
    if not isinstance(xyxy, np.ndarray): xyxy = np.asarray(xyxy)
    cxcywh = xyxy.copy()
    if fixed:
        cxcywh[:, 0] = (xyxy[:, 2] + xyxy[:, 0]) / 2  # cx
        cxcywh[:, 1] = (xyxy[:, 3] + xyxy[:, 1]) / 2  # cy
        dw = (xyxy[:, 2] - xyxy[:, 0]) * (scale[0] - 1)
        dh = (xyxy[:, 3] - xyxy[:, 1]) * (scale[1] - 1)
        dp = np.vstack((dw, dh))
        dp = np.max(dp, axis=0) if use_max else np.min(dp, axis=0)
        cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) + dp  # w
        cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) + dp  # h
    else:
        cxcywh[:, 0] = (xyxy[:, 2] + xyxy[:, 0]) / 2  # cx
        cxcywh[:, 1] = (xyxy[:, 3] + xyxy[:, 1]) / 2  # cy
        cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) * scale[0]  # w
        cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) * scale[1]  # h
    dxyxy = cxcywh2xyxy(cxcywh, width=None, height=None, normalized=False)
    if valid_range: dxyxy = clip_xyxy(dxyxy, valid_range=valid_range)
    return dxyxy


def extend_xywh(xywh: np.ndarray, scale=[1.0, 1.0], valid_range=[], fixed=False, use_max=True):
    """
    :param bboxes: [[xmin, ymin, xmax, ymax]]
    :param scale: [sx,sy]==>(W,H)
    :return:
    """
    if len(xywh) == 0 or len(scale) == 0: return xywh
    if not isinstance(xywh, np.ndarray): xywh = np.asarray(xywh)
    xyxy = xywh2xyxy(xywh)
    xyxy = extend_xyxy(xyxy, scale, valid_range=valid_range, fixed=fixed, use_max=use_max)
    dxywh = xyxy2xywh(xyxy)
    return dxywh


def extend_xyxy_similar_square(xyxy, use_max=True, weight=1.0, valid_range=[]):
    """
    将boxes扩展为近似正方形的boxes
    :param xyxy:
    :param use_max: 是否按照每个box(w,h)最大值进行转换
    :param weight: (0,1.0),weight越接近1.0，boxes越近似正方形
                   当weight=1.0，相当于get_square_boxes
                   当weight=0.0，相当于原始boxes
    :param valid_range:有效范围(xmin,ymin,xmax,ymax)
    :return:
    """
    if len(xyxy) == 0: return xyxy
    if not isinstance(xyxy, np.ndarray): xyxy = np.asarray(xyxy)
    center = xyxy2cxcywh(xyxy)
    if use_max:
        b = np.max(center[:, 2:4], axis=1)
    else:
        b = np.min(center[:, 2:4], axis=1)
    cw = b * weight + center[:, 2] * (1 - weight)
    ch = b * weight + center[:, 3] * (1 - weight)
    center[:, 2] = cw
    center[:, 3] = ch
    _boxes = cxcywh2xyxy(center)
    if valid_range: _boxes = clip_xyxy(_boxes, valid_range=valid_range)
    return _boxes


def shrink_polygon_pyclipper(polygon, ratio):
    """
    使用Polygon库计算多边形区域的周长和面积，使用pyclipper库进行shrink
    https://blog.csdn.net/shyjhyp11/article/details/126396170
    https://www.cnblogs.com/01black-white/p/15292193.html
    :param polygon: 
    :param ratio: 
    :return: 
    """
    from shapely.geometry import Polygon
    import pyclipper
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    # padding.MiterLimit = 100 # 多边形的尖角是否用圆角代替
    # padding.AddPath(subject, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


def get_square_boxes(boxes, use_max=True, use_mean=False, baseline=-1):
    """
    将boxes转换为正方形的boxes
    :param boxes:
    :param use_max: 是否按照每个box(w,h)最大值(True)/最小值(False)进行转换(默认)
    :param use_mean: 是否按照每个box(w,h)平均值进行转换(优先级比use_mean高)
    :param baseline: 当baseline>0，表示正方形最小边长
    :return:
    """
    if len(boxes) == 0: return boxes
    if not isinstance(boxes, np.ndarray): boxes = np.asarray(boxes)
    center = xyxy2cxcywh(boxes)
    if use_max:
        b = np.max(center[:, 2:4], axis=1)
    else:
        b = np.min(center[:, 2:4], axis=1)
    if use_mean:
        b = np.mean(center[:, 2:4], axis=1)
    if baseline > 0:
        index = b < baseline
        b[index] = baseline
    center[:, 2] = b
    center[:, 3] = b
    _boxes = cxcywh2xyxy(center)
    return _boxes


get_square_bboxes = get_square_boxes


def get_square_rects(rects, use_max=True, baseline=-1):
    """
    将rects转换为正方形的boxes
    :param rects: xywh
    :param use_max: 是否按照每个box(w,h)最大值进行转换
    :param baseline: 当baseline>0，表示正方形最小边长
    :return:
    """
    if len(rects) == 0: return rects
    boxes = xywh2xyxy(rects)
    boxes = get_square_boxes(boxes, use_max=use_max, baseline=baseline)
    rects = xyxy2xywh(boxes)
    return rects


def get_section(start, end, nums=2, scale=1.0, dtype=None):
    """
    均匀划分nums个线段，并返回截断点(nums+1)
    s1 = get_section(start, end, nums=nums)
    等价于：
    s2 = np.linspace(start, end, num=nums + 1,endpoint=True)
    :param start: 起点
    :param end: 终点
    :param nums: 将范围[start,end]均匀划分的段数，默认2段
    :param scale: 对范围[start,end]进行缩放
    :param dtype: 输出类型
    :return: 返回截断点,其个数是nums+1
    """
    d = end - start
    c = (start + end) * 0.5  # 中心点
    start = int(c - d * scale * 0.5)
    end = int(c + d * scale * 0.5)
    unit = (end - start) / nums
    out = np.arange(start, end + 0.1 * unit, unit)
    dtype = dtype if dtype else out.dtype
    out = np.asarray(out, dtype=dtype)
    return out


def get_boxes_iou(box1, box2):
    """
    :param box1: 预测框(n, 4)
    :param box2: GT框 (m, 4)
    :return: IOU (n, m)
    numpy 广播机制 从后向前对齐。 维度为1 的可以重复等价为任意维度
    eg: (4,3,2)   (3,2)  (3,2)会扩充为(4,3,2)
        (4,1,2)   (3,2) (4,1,2) 扩充为(4, 3, 2)  (3, 2)扩充为(4, 3,2) 扩充的方法为重复
    广播会在numpy的函数 如sum, maximun等函数中进行
    pytorch同理。
    扩充维度的方法：
    eg: a  a.shape: (3,2)  a[:, None, :] a.shape: (3, 1, 2) None 对应的维度相当于newaxis
    """
    if not isinstance(box1, np.ndarray):
        box1 = np.array(box1)
    if not isinstance(box2, np.ndarray):
        box2 = np.array(box2)
    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # left_top (x, y)
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # right_bottom (x, y)
    wh = np.maximum(rb - lt + 1, 0)  # inter_area (w, h)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]  # shape: (n, m)
    box_areas = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    gt_areas = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
    iou = inter_areas / (box_areas[:, None] + gt_areas - inter_areas)
    return iou


box_iou_v1 = get_boxes_iou


def box_iou_v2(box1, box2):
    """
    https://www.jb51.net/article/178732.htm
    :param box1:
    :param box2:
    :return:
    """
    if not isinstance(box1, np.ndarray):
        box1 = np.array(box1)
    if not isinstance(box2, np.ndarray):
        box2 = np.array(box2)
    xmin1, ymin1, xmax1, ymax1, = np.split(box1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(box2, 4, axis=-1)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    iou = intersect / union
    return iou


def get_box_iou(box1, box2):
    """
    计算IOU=交集(A,B)/并集(A,B)
    计算IOM=交集(A,B)/最小集(A,B)
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    if len(box1) == 0: return 0
    if len(box2) == 0: return 0
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


def get_box_iom(box1, box2):
    """
    计算IOU=交集(A,B)/并集(A,B)
    计算IOM=交集(A,B)/最小集(A,B)
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    if len(box1) == 0: return 0
    if len(box2) == 0: return 0
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / min(s1, s2)
    return iou


class YOLOCoords(object):
    def __init__(self, max_boxes=120, norm=False):
        self.max_boxes = max_boxes
        self.norm = norm

    def __call__(self, image, boxes, labels):
        dboxes = np.zeros((self.max_boxes, 4))
        dlabels = np.zeros((self.max_boxes))
        if len(boxes) > 0:
            width, height, _ = image.shape
            cboxes = xyxy2cxcywh(boxes, width, height, normalized=self.norm)
            dboxes[0:len(cboxes)] = cboxes
            dlabels[0:len(cboxes)] = labels
        return image, dboxes, dlabels


def show_image(name, image, boxes, labels, center2bboxes=False, untranspose=False, waitKey=0):
    from pybaseutils import image_utils
    if center2bboxes:
        boxes = cxcywh2xyxy(boxes)
    if untranspose:
        image = image_utils.untranspose(image)
    image = image_utils.show_image_boxes_texts(name, image, boxes, labels, delay=waitKey)
    return image


def demo_for_augment():
    from pybaseutils import image_utils
    input_size = [320, 320]
    image_path = "test.jpg"
    boxes = [[98, 42, 160, 100], [244, 260, 297, 332], [98 + 50, 42 + 50, 160 + 50, 100 + 50]]
    labels = [1, 2, 3]
    image = image_utils.read_image(image_path)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels)
    augment = YOLOCoords(max_boxes=120, norm=False)
    for i in range(1000):
        boxes = extend_xyxy(boxes, scale=[1.2, 1.2])
        dst_image, dst_label, dst_boxes = augment(image, boxes.copy(), labels.copy())
        image = show_image("detd", image, dst_label, dst_boxes, center2bboxes=True, untranspose=False, waitKey=0)


if __name__ == "__main__":
    start, end = 1, 20
    nums = 5
    s1 = get_section(start, end, nums=5)
    s2 = np.linspace(start, end, num=nums + 1, endpoint=True)
    print(s1)
    print(s2)
