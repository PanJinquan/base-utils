# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-05-12 08:54:44
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np


def annular_to_rect(image, center=None, inner_radius=0, outer_radius=None):
    """
    将圆环展开为矩形
    :param image:
    :param center:
    :param inner_radius:
    :param outer_radius:
    :return:
    """
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)  # 默认圆心
    if outer_radius is None:
        outer_radius = min(center[0], center[1], image.shape[1] - center[0], image.shape[0] - center[1])

    # 极坐标参数
    theta = np.linspace(0, 2 * np.pi, 360)  # 角度采样（宽度）
    r = np.arange(inner_radius, outer_radius)  # 半径采样（高度）

    # 生成极坐标网格
    r_grid, theta_grid = np.meshgrid(r, theta)

    # 转换为笛卡尔坐标
    x = center[0] + r_grid * np.cos(theta_grid)
    y = center[1] + r_grid * np.sin(theta_grid)

    # 双线性插值获取像素值
    rect_img = cv2.remap(image,
                         x.astype(np.float32),
                         y.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    return rect_img


# 示例用法
image_file = "/home/PKing/Downloads/mask/微信图片_2025-05-12_085307_786.png"
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # 读取圆形mask图
rect_mask = annular_to_rect(image, inner_radius=50, outer_radius=200)  # 设置内外半径
cv2.imshow("Rectangular Mask", rect_mask)
cv2.waitKey(0)