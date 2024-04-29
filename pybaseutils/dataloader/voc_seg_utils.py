# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-05 11:54:13
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
import PIL.Image


def change_format(contour):
    contour2 = []
    length = len(contour)
    for i in range(0, length, 2):
        contour2.append([contour[i], contour[i + 1]])
    return np.asarray(contour2, np.int32)


def get_segment_area(seg_path, bbox):
    """
    :param seg_path:
    :param bbox: bbox = [xmin, ymin, xmax, ymax]
    :return:stroke_segs is [[...],[...]],area is int
    """
    area = 0
    bbox = [int(b) for b in bbox]
    seg = read_segment_image(seg_path, bbox)
    if seg:
        seg = [list(map(float, seg))]
        contour = change_format(seg[0])
        # 计算轮廓面积
        area = abs(cv2.contourArea(contour, True))
    return seg, area


def read_segment_image(seg_file, bbox):
    """
    :param seg_file:
    :param bbox: bbox = [xmin, ymin, xmax, ymax]
    :return:
    """
    if not os.path.exists(seg_file):
        return []
    try:
        mask_1 = cv2.imread(seg_file, 0)
        mask = np.zeros_like(mask_1, np.uint8)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_1[bbox[1]:bbox[3],
                                                 bbox[0]:bbox[2]]

        # 计算矩形中点像素值
        mean_x = (bbox[0] + bbox[2]) // 2
        mean_y = (bbox[1] + bbox[3]) // 2

        end = min((mask.shape[1], int(bbox[2]) + 1))
        start = max((0, int(bbox[0]) - 1))

        flag = True
        for i in range(mean_x, end):
            x_ = i
            y_ = mean_y
            pixels = mask_1[y_, x_]
            if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                mask = (mask == pixels).astype(np.uint8)
                flag = False
                break
        if flag:
            for i in range(mean_x, start, -1):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:
                    mask = (mask == pixels).astype(np.uint8)
                    break
        polygons = mask2polygons(mask)
        return polygons
    except:
        return []


def mask2polygons(mask):
    '''从mask提取边界点'''
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
    bbox = []
    for cont in contours[0]:
        [bbox.append(i) for i in list(cont.flatten())]
        # map(bbox.append,list(cont.flatten()))
    return bbox  # list(contours[1][0].flatten())


def getbbox(height, width, points):
    '''边界点生成mask，从mask提取定位框'''
    # img = np.zeros([self.height,self.width],np.uint8)
    # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
    # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
    polygons = points
    mask = polygons_to_mask([height, width], polygons)
    return mask2box(mask)


def mask2box(mask):
    '''从mask反算出其边框
    mask：[h,w]  0、1组成的图片
    1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
    '''
    # np.where(mask==1)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    # 解析左上角行列号
    left_top_r = np.min(rows)  # y
    left_top_c = np.min(clos)  # x

    # 解析右下角行列号
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)

    # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
    # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
    # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
    return [left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式


def polygons_to_mask(img_shape, polygons):
    '''边界点生成mask'''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask
