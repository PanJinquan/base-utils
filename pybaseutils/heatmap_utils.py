# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-04-14 14:29:30
"""
import cv2
import numpy as np
import numpy.matlib


def create_gaussian_mask(width, height, center, radius=5):
    """
    生成二维高斯分布蒙版，width, height很大时，速度很慢
    :param width:
    :param height:
    :param center:
    :param radius: 高斯半径
    :return: gauss_mask : float32,(0,1)
    """
    # 利用for循环实现（很慢）
    # gauss_mask = np.zeros((height, width), dtype=np.float32)
    # for i in range(height):
    #     for j in range(width):
    #         dis = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
    #         gauss_mask[i, j] = np.exp(-0.5 * dis / radius)
    mask_x = np.matlib.repmat(center[0], height, width)
    mask_y = np.matlib.repmat(center[1], height, width)
    x1 = np.arange(width)
    x_map = np.matlib.repmat(x1, height, 1)
    y1 = np.arange(height)
    y_map = np.matlib.repmat(y1, width, 1)
    y_map = np.transpose(y_map)
    gauss_mask = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)
    gauss_mask = np.exp(-0.5 * gauss_mask / radius)
    # gauss_mask = np.clip(gauss_mask, 0, 1)
    return gauss_mask


def get_image_resize(image, coordinates, dst_wh=[448, 448]):
    height, width, d = image.shape
    scale = [dst_wh[0] / width, dst_wh[1] / height]
    image = cv2.resize(image, dsize=tuple(dst_wh))
    coordinates = [[int(c[0] * scale[0]), int(c[1] * scale[1])] for c in coordinates]
    return image, coordinates


def get_coordinates_resize(coordinate, src_wh, dst_wh=[448, 448]):
    height, width = src_wh
    scale = [dst_wh[0] / width, dst_wh[1] / height]
    # coordinates = [[int(c[0] * scale[0]), int(c[1] * scale[1])] for c in coordinates]
    coordinate = [int(coordinate[0] * scale[0]), int(coordinate[1] * scale[1])]
    return coordinate


def fast_create_gaussian_mask(width, height, center, radius=2):
    """
    快速生成二维高斯分布蒙版
    :param width:
    :param height:
    :param center:
    :param radius: 高斯半径
    :return:
    """
    dst_wh = [128, 128]
    scale = [dst_wh[0] / width, dst_wh[1] / height]
    gauss_mask = create_gaussian_mask(int(width * scale[0]),
                                      int(height * scale[1]),
                                      [int(center[0] * scale[0]), int(center[1] * scale[1])],
                                      radius=radius)
    gauss_mask = cv2.resize(gauss_mask, dsize=(width, height))
    return gauss_mask


def get_image_heatmap(image, points: list, input_size=[448, 448], radius=2, fusion="gray"):
    """
    生成图像的热力图
    :param image: image array
    :param points: 坐标列表(N,2)
    :param radius: 半径
    :param fusion: 融合方式:[color,white,black,split]
    :return:
    """
    # 生成热力图
    image, points = get_image_resize(image, points, dst_wh=input_size)
    h, w, d = image.shape
    mask = np.zeros((h, w), dtype=np.float32)
    for point in points:
        # mask += create_gaussian_mask(width=w, height=h, center=point, radius=radius)
        mask += fast_create_gaussian_mask(width=w, height=h, center=point, radius=radius)
    out_mask = np.clip(mask * 255.0, 0, 255)
    out_mask = np.asarray(out_mask, dtype=np.uint8)
    # 线性融合
    if fusion == "color":
        out_mask = cv2.applyColorMap(out_mask, cv2.COLORMAP_JET)
        heatmap = np.asarray(out_mask, dtype=np.float32)
        heatmap = heatmap * mask[:, :, np.newaxis]
        image = np.asarray(image, dtype=np.float32) + heatmap
        overlay = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    elif fusion == "white":
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2BGR)
        heatmap = np.asarray(out_mask, dtype=np.float32)
        image = np.asarray(image, dtype=np.float32) + heatmap
        overlay = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    elif fusion == "black":
        out_mask = cv2.cvtColor(255 - out_mask, cv2.COLOR_GRAY2BGR)
        heatmap = np.asarray(out_mask, dtype=np.float32) / 255
        image = np.asarray(image, dtype=np.float32) * heatmap
        overlay = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    elif fusion == "split":
        overlay = image.copy()
        overlay[:, :, 1] = out_mask
    else:
        raise Exception("fusion ERROR:{}".format(fusion))
    # dst = src1[i] * alpha + src2[i] * beta + gamma;//两张图片每个通道对应数值之和。
    # 根据mask进行融合
    # ret, bmask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    # img1_bg = cv2.bitwise_and(mask, mask, mask=bmask)
    # img1_bg = cv2.cvtColor(img1_bg, cv2.COLOR_GRAY2BGR)
    # overlay = cv2.add(img1_bg, image)  # 进行融合
    return overlay, out_mask


if __name__ == "__main__":
    from pybaseutils import image_utils

    image_path = "/home/dm/project/python-learning-notes/dataset/test_image/test1.jpg"
    image = image_utils.read_image(image_path)
    input_size = [416, 416]
    points = [[100, 200], [400, 200]]
    colormap = "split"
    # image = np.zeros_like(image, dtype=np.uint8)
    image, mask = get_image_heatmap(image, points, input_size=input_size, fusion=colormap)
    image_utils.cv_show_image("mask", mask, delay=1)
    image_utils.cv_show_image("image", image)
