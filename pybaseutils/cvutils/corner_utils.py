# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-24 22:20:14
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import image_utils, file_utils
from pybaseutils.cluster import kmean


def get_order_points(pts_src):
    """
    对4个点进行排序:[top-left, top-right, bottom-right, bottom-left]
    top-left    ：对应y+x之和的最小点
    bottom-right：对应y+x之和的最大点
    top-right   ：对应y-x之差的最小点
    bottom-left ：对应y-x之差的最大点
    :param pts_src: pts_dst [top-left, top-right, bottom-right, bottom-left]
    :return:
    """
    pts_src = np.array(pts_src)
    pts_dst = np.zeros_like(pts_src)
    s = pts_src.sum(axis=1)
    # Top-left point will have the smallest sum.
    pts_dst[0] = pts_src[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    pts_dst[2] = pts_src[np.argmax(s)]
    diff = np.diff(pts_src, axis=1)  # y-x= pts_src[:, 1] - pts_src[:, 0]
    # Top-right point will have the smallest difference.
    pts_dst[1] = pts_src[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    pts_dst[3] = pts_src[np.argmax(diff)]
    # Return the ordered coordinates.
    return pts_dst


def get_image_four_corners(image, n_corners=4, ksize=5, blur=True, max_iter=10, vis=False):
    """
    获得图像的角点
       findContours的用法：https://blog.csdn.net/xfijun/article/details/117694917
       approxPolyDP的用法：http://t.zoukankan.com/bjxqmy-p-12347265.html
    :param image: 输入BGR图像
    :param n_corners: 最多角点的个数
    :param ksize:
    :param blur: 边缘检测前，进行模糊可以有效去除虚假边缘的影响
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image
    # 高斯滤波去除图像噪声
    if blur: gray = cv2.GaussianBlur(gray, ksize=(2 * ksize + 1, 2 * ksize + 1), sigmaX=0)
    # if blur: gray = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=0)
    # 进行边缘检测
    canny = cv2.Canny(gray, threshold1=0, threshold2=200)
    # 进行膨胀，合并边缘两侧
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    # 查询边缘的所有轮廓
    contours, hierarchy = cv2.findContours(canny, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return np.zeros((0, 2), np.int32)
    if vis: cv2.drawContours(image, contours, -1, color=(255, 0, 0), thickness=1)
    # Keeping only the largest detected contour.
    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]
    if vis: cv2.drawContours(image, [contour], -1, color=(0, 255, 0), thickness=1)
    xmin, ymin, xmax, ymax = [min(contour[:, 0, 0]), min(contour[:, 0, 1]),
                              max(contour[:, 0, 0]), max(contour[:, 0, 1])]
    if vis: cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
    # 使用approxPolyDP拟合多边形，减少多边形点的个数
    # max_dist = 0.02 * cv2.arcLength(contour, closed=True)  # 轮廓周长
    max_dist = 0.3 * sum([xmax - xmin, ymax - ymin]) / 2  # 外接矩形框
    corners = cv2.approxPolyDP(contour, max_dist, closed=True)  # 所有角点
    corners = corners.reshape(-1, 2)
    if vis: print("contour={},corners={},max_dist={}".format(len(contour), len(corners), max_dist))
    # 如果不够四个角点，则减小max_dist
    decay = 0.8
    iter = 0
    while len(corners) < n_corners and iter < max_iter:
        iter += 1
        max_dist = decay * max_dist
        corners = cv2.approxPolyDP(contour, max_dist, closed=True)  # 所有角点
        corners = corners.reshape(-1, 2)
        if vis: print("iter={},contour={},corners={},max_dist={}".format(iter, len(contour), len(corners), max_dist))
    # 如果角点大于4个，则进行kmeans聚类
    if len(corners) > n_corners:
        index = kmean.sklearn_kmeans(corners, n_clusters=n_corners, max_iter=max_iter)
        clusters = []
        for i in range(n_corners):
            c = np.mean(corners[index == i], axis=0)  # 取中心点
            clusters.append(c)
        corners = clusters
    # Sorting the corners and converting them to desired shape.
    corners = get_order_points(np.asarray(corners))
    if vis:
        image = image_utils.draw_image_points_lines(image, corners, fontScale=0.8, thickness=2)
        image_utils.cv_show_image("canny-image", image, use_rgb=False)
    return corners


def get_document_corners_grabcut(image, ksize=5, blur=True):
    """
    https://colorspace.blog.csdn.net/article/details/126111605
    使用grabcut工具文档区域的四个角点
    :param image: 输入图像
    :param ksize: 形态学处理大小
    :param blur: 是否进行模糊
    :return:
    """
    # 闭运算
    kernel = np.ones((ksize, ksize), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    # 用GrabCut去掉背景
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    corners = get_image_four_corners(image, ksize=ksize, blur=blur)
    return corners


def get_document_corners_simple(image, ksize=5, blur=True):
    """
    获取文档区域的四个角点
    :param image: 输入图像
    :param ksize: 形态学处理大小
    :param blur: 是否进行模糊
    :return:
    """
    # 闭运算
    kernel = np.ones((ksize, ksize), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    corners = get_image_four_corners(image, ksize=ksize, blur=blur)
    return corners


def get_document_corners(image, target=320, ksize=5, blur=True, grabcut=False):
    """
    获取文档区域的四个角点
    :param image: 输入图像
    :param target: 处理图像大小，值越大精度越高，但速度越慢
    :param ksize: 形态学处理大小
    :param blur: 是否进行模糊
    :param grabcut: 是否使用grabcut工具
    :return:
    """
    h1, w1 = image.shape[:2]
    input = image
    if (h1 > target or w1 > target) and target > 0:
        input = image_utils.resize_image(input, size=(target, None))
    h2, w2 = input.shape[:2]
    if grabcut:
        corners = get_document_corners_grabcut(input, ksize=ksize, blur=blur)
    else:
        corners = get_document_corners_simple(input, ksize=ksize, blur=blur)
    corners = corners * (w1 / w2, h1 / h2)
    return corners


def get_document_corners_example(image_dir):
    """
    获取文档区域的四个角点demo
    :param image_dir:
    :return:
    """
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        image_file = "/home/dm/nasdata/dataset/csdn/文档矫正/image2/test04.jpg"
        print(image_file)
        image = cv2.imread(image_file)
        corners = get_document_corners(image)
        image = image_utils.draw_image_points_lines(image, corners, fontScale=2.0, thickness=4)
        image_utils.cv_show_image("image", image, use_rgb=False)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/文档矫正/image2"
    get_document_corners_example(image_dir)
