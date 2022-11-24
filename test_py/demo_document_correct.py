# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import mouse_utils, file_utils, image_utils


def get_target_points(src_pts: np.ndarray):
    """
    根据输入的四个角点，计算其矫正后的目标四个角点,src_pts四个点分布：
        0--(w01)---1
        |          |
      (h03)      (h21)
        |          |
        3--(w23)---2
    :param src_pts:
    :return:
    """
    # 计算四个角点的边长
    w01 = np.sum(np.square(src_pts[0] - src_pts[1]), axis=0)
    h03 = np.sum(np.square(src_pts[0] - src_pts[3]), axis=0)
    h21 = np.sum(np.square(src_pts[2] - src_pts[1]), axis=0)
    w23 = np.sum(np.square(src_pts[2] - src_pts[3]), axis=0)
    xmin, ymin = 0, 0
    if h03 > w01:
        xmax = np.sqrt(np.mean([w01, w23]))
        ymax = np.sqrt(np.mean([h03, h21]))
    else:
        xmax = np.sqrt(np.mean([w01, w23]))
        ymax = np.sqrt(np.mean([h03, h21]))
    # print("h/w:{}".format(abs(23.4 / 17.8 - ymax / xmax)))
    dst_pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    dst_pts = np.asarray(dst_pts)
    return dst_pts


def document_image_correct(src, src_pts, dst_pts=None, out_size=None, use_ransac=False, vis=False):
    """
    getPerspectiveTransform使用4种对应关系(这是计算单应性/透视变换的最低要求)来计算变换，
    其中findHomography即使您提供了4种以上的对应关系(大概使用某种东西)也可以计算变换
    :param src: 输入原始图像
    :param src_pts: 原始图像的四个角点
    :param dst_pts: 目标图像的四个角点,默认为None，表示自动获取dst_pts
    :param out_size: 目标图像输出大小，默认为None，表示与dst_pts相同大小
    :param use_ransac: 是否使用findHomography估计变换矩阵
    :param vis: 可视化效果
    :return:
    """
    src_pts = np.float32(src_pts)
    if dst_pts is None: dst_pts = get_target_points(src_pts)
    dst_pts = np.float32(dst_pts)
    if out_size is None:
        # xmin = min(pts_dst[:, 0])
        # ymin = min(pts_dst[:, 1])
        xmax = int(max(dst_pts[:, 0]))
        ymax = int(max(dst_pts[:, 1]))
        out_size = [xmax, ymax]
    if use_ransac:
        m, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    else:
        m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst = cv2.warpPerspective(src, m, dsize=out_size, flags=cv2.INTER_LINEAR)
    if vis:
        dst = image_utils.draw_points_text(dst, np.int32(dst_pts), color=(0, 255, 0), thickness=3)
        image_utils.cv_show_image("correct", dst, use_rgb=False)
    return dst


def document_correct_image_example(image, winname="document_correct"):
    pts_src = np.zeros(shape=(0, 2), dtype=np.int32)
    mouse = mouse_utils.DrawImageMouse(max_point=4, thickness=3)
    while len(pts_src) < 4:
        pts_src = mouse.draw_image_polygon_on_mouse(image, winname=winname)
        # pts_src = [[345., 114.], [579., 349.], [346., 582.], [111., 347.]]
        pts_src = np.asarray(pts_src)
        if len(pts_src) < 4:
            mouse.clear()
            print("已经标记了文档的{}个角点，需要标记4个角点".format(len(pts_src)))
            cv2.waitKey(0)
    print("标记文档的4个角点={}".format(pts_src.tolist()))
    image = image_utils.draw_image_points_lines(image, pts_src, thickness=2)
    image_utils.cv_show_image(winname, image, use_rgb=False, delay=10)
    document_image_correct(image, pts_src, vis=True)


if __name__ == '__main__':
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/page-correct1"
    # image_dir = "/home/dm/nasdata/dataset/csdn/文档矫正"
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        image = cv2.imread(image_file)
        document_correct_image_example(image)
    cv2.waitKey(0)
