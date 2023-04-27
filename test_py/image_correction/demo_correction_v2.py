# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils.cvutils import mouse_utils, corner_utils
from pybaseutils import file_utils, image_utils


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
    if len(src_pts) != 4:
        print("输入src_pts必须含有4个角点:{}".format(src_pts))
        return src
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
        image_utils.cv_show_image("correct-result", dst, use_rgb=False)
    return dst


def document_correct_by_mouse(image, winname="document_correct_by_mouse"):
    """
    通过鼠标操作获得文档的四个角点
    :param image: 输入图像
    :param winname: 窗口名称
    :return:
    """
    corners = np.zeros(shape=(0, 2), dtype=np.int32)
    mouse = mouse_utils.DrawImageMouse(max_point=4, thickness=5)
    # image_utils.cv_show_image("correct-result", np.zeros_like(image) + 0, use_rgb=False, delay=1)
    while len(corners) < 4:
        corners = mouse.draw_image_polygon_on_mouse(image, winname=winname)
        corners = np.asarray(corners)
        if len(corners) < 4:
            mouse.clear()
            print("已经标记了文档的{}个角点，需要标记4个角点".format(len(corners)))
            cv2.waitKey(0)
    print("标记文档的4个角点={}".format(corners.tolist()))
    return corners


def document_correct_by_auto(image, winname="document_correct_by_auto", vis=False):
    """
    通过算法自动获得文档的四个角点
    :param image: 输入图像
    :param winname: 窗口名称
    :param vis: 是否可视化
    :return:
    """
    corners = corner_utils.get_document_corners(image)
    if vis:
        image = image_utils.draw_image_points_lines(image, corners, fontScale=2.0, thickness=5)
        image_utils.cv_show_image(winname, image, use_rgb=False)
    return corners


def document_correct_image_example(image, use_mouse=False, winname="document", vis=True):
    """
    通过算法自动获得文档的四个角点
    :param image: 输入图像
    :param use_mouse: True:通过鼠标操作获得文档的四个角点
                      False:通过算法自动获得文档的四个角点
    :param winname: 窗口名称
    :param vis: 可视化效果
    :return:
    """
    # 获得文档的四个角点
    if use_mouse:
        corners = document_correct_by_mouse(image, winname=winname)  # 通过鼠标操作获得文档的四个角点;
    else:
        corners = document_correct_by_auto(image)  # 通过算法自动获得文档的四个角点
    # 在原图显示角点
    image = image_utils.draw_image_points_lines(image, corners, circle_color=(0, 255, 0), fontScale=2.0, thickness=5)
    image_utils.cv_show_image(winname, image, use_rgb=False, delay=10)
    # 实现文档矫正
    document_image_correct(image, corners, vis=vis)


if __name__ == '__main__':
    # image_dir = "data/image2"  # 测试图片
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/page-correct/image3"  # 测试图片
    use_mouse = True  # 是否通过鼠标操作获得文档的四个角点
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        print(image_file)
        image = cv2.imread(image_file)
        document_correct_image_example(image, use_mouse=use_mouse)
    cv2.waitKey(0)
