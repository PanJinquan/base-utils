# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-12-14 15:09:34
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils.transforms.transform_utils import *


def get_reference_facial_points(out_size=(112, 112), square=True, vis=False):
    """
    获得人脸参考关键点,目前支持两种输入的参考关键点,即[96, 112]和[112, 112]
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    ==================
    face_size_ref = [112, 112]
    kpts_ref = [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]

    ==================
    square = True, crop_size = (112, 112)
    square = False,crop_size = (96, 112),
    :param square: True is [112, 112] or False is [96, 112]
    :param vis   : True or False,是否显示
    :return:
    """
    size_ref = (96, 112)
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    dst_pts = np.asarray(kpts_ref, dtype=np.float32)
    dst_size = size_ref
    if square or out_size[0] != size_ref[0] or out_size[1] != size_ref[1]:
        size_ref = np.array(size_ref, dtype=np.float32)
        maxL = max(size_ref)
        wh_diff = maxL - size_ref
        dst_pts = dst_pts + wh_diff / 2.0
        dst_pts = dst_pts * out_size / maxL
        dst_size = size_ref + wh_diff
        dst_size = dst_size * out_size / maxL
    dst_size = (int(dst_size[0]), int(dst_size[1]))
    if vis:
        from pybaseutils import image_utils
        tmp = np.zeros(shape=(int(dst_size[1]), int(dst_size[0]), 3), dtype=np.uint8)
        tmp = image_utils.draw_landmark(tmp, [dst_pts], vis_id=True)
        cv2.imshow("kpts_ref", tmp)
        cv2.waitKey(0)
    return dst_pts


def get_facial_points(out_size=(112, 112), extend=(1.0, 1.0), square=True, vis=False):
    """
    :param out_size:
    :param extend:
    :param square:
    :param vis:
    :return:
    """
    dst_pts = get_reference_facial_points(out_size=out_size, square=square, vis=False)
    dst_pts, out_size = extend_facial_points(dst_pts, out_size=out_size, extend=extend, vis=vis)
    return dst_pts, out_size


def extend_facial_points(src_pts, out_size=(112, 112), extend=(1.5, 1.5), vis=False):
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_size = np.array(out_size, dtype=np.float32)
    dst_size = dst_size * extend
    wh_diff = dst_size - out_size
    dst_pts = src_pts + wh_diff / 2.0
    if vis:
        from pybaseutils import image_utils
        tmp = np.zeros(shape=(int(dst_size[1]), int(dst_size[0]), 3), dtype=np.uint8)
        tmp = image_utils.draw_landmark(tmp, [dst_pts], vis_id=True)
        cv2.imshow("kpts_ref", tmp)
        cv2.waitKey(0)
    dst_size = (int(dst_size[0]), int(dst_size[1]))
    return dst_pts, dst_size


def face_alignment(image, src_pts, out_size=(112, 112), extend=(), method="lstsq"):
    """
    实现人脸校准
    :param image: input image
    :param src_pts: 原始点S集合(n×2)
    :param out_size: 变换后输出图像大小
    :param extend: 裁剪缩放大小
    :param method: lstsq,estimate,affine,homo
    :return:  align_image 对齐后的图像
              M           S->D的变换矩阵(2×3)
              Minv        D->S的逆变换矩阵(2×3)
    """
    if extend:
        dst_pts, out_size = get_facial_points(out_size=out_size, extend=extend, square=True, vis=False)
        align_face, M, Minv = get_image_alignment(image, src_pts, dst_pts, out_size, method="lstsq")
    else:
        # 获得标准人脸关键点
        dst_pts = get_reference_facial_points(out_size=out_size, square=True, vis=False)
        align_face, M, Minv = get_image_alignment(image, src_pts, dst_pts, out_size, method=method)
    return align_face, M, Minv


if __name__ == "__main__":
    from pybaseutils import image_utils

    image_file = "test.jpg"
    out_size = (112, 112)
    image = cv2.imread(image_file)
    # face detection from MTCNN
    boxes = np.asarray([[200.27724761, 148.9578526, 456.70521605, 473.52968433]])
    src_pts = np.asarray([[[287.86636353, 306.13598633],
                           [399.58618164, 272.68032837],
                           [374.80252075, 360.95596313],
                           [326.71264648, 409.12332153],
                           [419.06210327, 381.41421509]]])

    src_pts = np.asarray(src_pts, dtype=np.float32)
    align_image, M, Minv = face_alignment(image, src_pts, out_size, method="lstsq", extend=(1.5, 1.5))
    print("M   :\n", M)
    print("Minv:\n", Minv)
    print("align_image:\n", align_image.shape)
    image_utils.cv_show_image("image", image, delay=10)
    image_utils.cv_show_image("align_image", align_image)
