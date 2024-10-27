# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-02-14 15:34:50
"""

import os
import copy
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import matplotlib
import numbers
import base64
import math
import platform
import PIL.Image as Image
from typing import List, Dict, Tuple
from PIL import ImageDraw, ImageFont
from math import cos, sin
from pybaseutils import file_utils, color_utils
from pybaseutils.coords_utils import *
from pybaseutils.cvutils import corner_utils

color_table = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
               (128, 0, 0), (0, 128, 0), (128, 128, 0),
               (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
               (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
               (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)] * 100

color_map = [(0, 0, 0), (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
             (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
             (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
             (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)] * 100

root = os.path.dirname(__file__)
coco_skeleton = [[1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8],
                 [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
coco_skeleton_v2 = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                    [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
mpii_skeleton = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [6, 3], [12, 11], [7, 12],
                 [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [7, 13]]


def get_font_type(size, font=""):
    """
    Windows字体路径      : /usr/share/fonts/楷体.ttf
    Linux(Ubuntu)字体路径：/usr/share/fonts/楷体.ttf
     >> fc-list             查看所有的字体
     >> fc-list :lang=zh    查看所有的中文字体
    :param size: 字体大小
    :param font:  simsun.ttc 宋体;simhei.ttf 黑体
    :return:
    """
    # 参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：
    if font:
        font = ImageFont.truetype(font, size, encoding="utf-8")
    elif platform.system().lower() == 'windows':
        font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")  # simsun.ttc 宋体
    elif platform.system().lower() == 'linux':
        # font = ImageFont.truetype("uming.ttc", size, encoding="utf-8")
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size, encoding="utf-8")
    else:
        font = ImageFont.truetype(os.path.join(root, "font_style/simhei.ttf"), size, encoding="utf-8")
    return font


def create_image(shape, color=(255, 255, 255), dtype=np.uint8, use_rgb=False):
    """
    生成一张图片
    :param shape:
    :param color: (b,g,r)
    :param dtype:
    :param use_rgb:
    :return:
    """
    image = np.zeros(shape, dtype=np.uint8)
    if image.ndim == 2: return np.asarray(image + color[0], dtype=dtype)
    for i in range(len(color)):
        image[:, :, i] = color[i]
    if use_rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def points_protection(points, height, width):
    """
    Avoid array overbounds
    :param points:
    :param height:
    :param width:
    :return:
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    # points[points[:, 0] > width, 0] = width - 1  # x
    # points[points[:, 1] > height, 1] = height - 1  # y
    # points[points[:, 0] < 0, 0] = 0  # x
    # points[points[:, 1] < 0, 1] = 0  # y
    if len(points) > 0:
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def boxes_protection(boxes, width, height):
    """
    :param boxes:
    :param width:
    :param height:
    :return:
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)
    if len(boxes) > 0:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def tensor2image(batch_tensor, index=0):
    """
    convert tensor to image
    :param batch_tensor:
    :param index:
    :return:
    """
    image_tensor = batch_tensor[index, :]
    image = np.array(image_tensor, dtype=np.float32)
    image = np.squeeze(image)
    image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    return image


def get_image_tensor(image_path, image_size, transpose=False):
    image = read_image(image_path)
    # transform = default_transform(image_size)
    # torch_image = transform(image).detach().numpy()
    image = resize_image(image, size=(int(128 * image_size[0] / 112), int(128 * image_size[1] / 112)))
    image = center_crop(image, crop_size=image_size)
    image_tensor = image_normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if transpose:
        image_tensor = image_tensor.transpose(2, 0, 1)  # NHWC->NCHW
    image_tensor = image_tensor[np.newaxis, :]
    # std = np.std(torch_image-image_tensor)
    return image_tensor


def image_clip(image):
    """
    :param image:
    :return:
    """
    image = np.clip(image, 0, 1)
    return image


def check_point(point):
    """检测点知否在图像内"""
    if point is None: return False
    if len(point) == 0: return False
    r = point[0] > 0 and point[1] > 0
    return r


def transpose(data):
    data = data.transpose(2, 0, 1)  # HWC->CHW
    return data


def untranspose(data):
    if len(data.shape) == 3:
        data = data.transpose(1, 2, 0).copy()  # 通道由[c,h,w]->[h,w,c]
    else:
        data = data.transpose(1, 0).copy()
    return data


def swap_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image[:, :, ::-1]  # RGB->BGR
    return image


def show_batch_image(title, batch_images, index=0):
    '''
    批量显示图片
    :param title:
    :param batch_images:
    :param index:
    :return:
    '''
    image = batch_images[index, :]
    # image = image.numpy()  #
    image = np.array(image, dtype=np.float32)
    image = np.squeeze(image)
    image = untranspose(image)
    if title:
        cv_show_image(title, image)


def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    channel = len(image.shape)
    if channel == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def cv_show_image(title, image, use_rgb=False, delay=0):
    """
    调用OpenCV显示图片
    :param title: 图像标题
    :param image: 输入是否是RGB图像
    :param use_rgb: True:输入image是RGB的图像, False:返输入image是BGR格式的图像
    :param delay: delay=0表示暂停，delay>0表示延时delay毫米
    :return:
    """
    img = image.copy()
    if img.shape[-1] == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    # cv2.namedWindow(title, flags=cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(delay)
    return img


def show_images_list(name, images_list, delay=0):
    out = image_hstack(images_list)
    cv2.imshow(name, out)
    cv2.waitKey(delay)


def resize_image_like(image_list, dst_img, is_rgb=False, use_pad=False, interpolation=cv2.INTER_NEAREST):
    """
    按dst_img的图像大小对image_list所有图片进行resize
    :param image_list: 图片列表
    :param dst_img: 目标图片大小
    :param is_rgb: 是否将灰度图转换为RGB格式
    :return:
    """
    shape = dst_img.shape
    is_rgb = len(shape) == 3 or is_rgb
    for i in range(len(image_list)):
        if not shape[:2] == image_list[i].shape[:2]:
            if use_pad:
                image_list[i] = resize_image_padding(image_list[i], (shape[1], shape[0]), interpolation=interpolation)
            else:
                image_list[i] = cv2.resize(image_list[i], dsize=(shape[1], shape[0]), interpolation=interpolation)
        if is_rgb and len(image_list[i].shape) == 2:
            image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_GRAY2BGR)
    return image_list


def image_hstack(images, split_line=False, is_rgb=False, texts=[], fontScale=-1.0, thickness=-1, use_pad=False):
    """图像左右拼接"""
    if len(images) == 0: return images
    dst_images = resize_image_like(image_list=images, dst_img=images[0], is_rgb=is_rgb, use_pad=use_pad)
    thickness, fontScale = get_linesize(max(images[0].shape), thickness=thickness, fontScale=fontScale)
    dst_images = np.hstack(dst_images)
    if len(dst_images.shape) == 2:
        dst_images = cv2.cvtColor(dst_images, cv2.COLOR_GRAY2BGR)
    if split_line or texts:
        h, w = dst_images.shape[:2]
        x = w // len(images)
        y = h
        for i in range(len(images)):
            p1 = (i * x, 0)
            p2 = (i * x, y)
            if split_line: dst_images = cv2.line(dst_images, p1, p2, color=(255, 0, 0), thickness=thickness)
            if texts: dst_images = draw_text(dst_images, point=(i * x + 10, 20 * thickness), color=(0, 0, 255),
                                             fontScale=fontScale, thickness=thickness, text=texts[i], drawType="simple")
    return dst_images


def image_vstack(images, split_line=False, is_rgb=False, texts=[], fontScale=-1.0, thickness=-1, use_pad=False):
    """图像上下拼接"""
    if len(images) == 0: return images
    dst_images = resize_image_like(image_list=images, dst_img=images[0], is_rgb=is_rgb, use_pad=use_pad)
    thickness, fontScale = get_linesize(max(images[0].shape), thickness=thickness, fontScale=fontScale)
    dst_images = np.vstack(dst_images)
    if len(dst_images.shape) == 2:
        dst_images = cv2.cvtColor(dst_images, cv2.COLOR_GRAY2BGR)
    if split_line or texts:
        h, w = dst_images.shape[:2]
        x = w
        y = h // len(images)
        for i in range(len(images)):
            p1 = (0, i * y)
            p2 = (x, i * y)
            if split_line: dst_images = cv2.line(dst_images, p1, p2, color=(255, 0, 0), thickness=2)
            if texts: dst_images = draw_text(dst_images, point=(10, i * y + 30),
                                             fontScale=1.0, thickness=2, text=texts[i], drawType="simple")
    return dst_images


def image_fliplr(image):
    '''
    左右翻转
    :param image:
    :return:
    '''
    image = np.fliplr(image)
    return image


def get_prewhiten_image(x):
    '''
    图片白化处理
    :param x:
    :return:
    '''
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def image_normalize(image, mean=None, std=None):
    '''
    正则化，归一化
    image[channel] = (image[channel] - mean[channel]) / std[channel]
    :param image: numpy image
    :param mean: [0.5,0.5,0.5]
    :param std:  [0.5,0.5,0.5]
    :return:
    '''
    # 不能写成:image=image/255
    if isinstance(mean, list):
        mean = np.asarray(mean, dtype=np.float32)
    if isinstance(std, list):
        std = np.asarray(std, dtype=np.float32)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    if mean is not None:
        image = np.subtract(image, mean)
    if std is not None:
        image = np.multiply(image, 1 / std)
    return image


def image_unnormalize(image, mean=None, std=None):
    """
    正则化，归一化
    image = image/255.0
    image[channel] = (image[channel] - mean[channel]) / std[channel]
    :param image: numpy image
    :param mean: [0.5,0.5,0.5]
    :param std:  [0.5,0.5,0.5]
    :return:
    """
    # 不能写成:image=image/255
    if isinstance(mean, list):
        mean = np.asarray(mean, dtype=np.float32)
    if isinstance(std, list):
        std = np.asarray(std, dtype=np.float32)
    image = np.multiply(image, std)
    image = (image + mean) * 255
    image = np.clip(image, 0, 255)
    image = np.array(image, dtype=np.uint8)
    return image


image_unnormalization = image_unnormalize


def data_normalize(x, ymin=0, ymax=1.0, xmin=None, xmax=None):
    """
    NORMALIZATION 将数据x归一化到任意区间[ymin,ymax]范围的方法
    :param x:  输入参数x：需要被归一化的数据,numpy
    :param ymin: 输入参数ymin：归一化的区间[ymin,ymax]下限
    :param ymax: 输入参数ymax：归一化的区间[ymin,ymax]上限
    :param xmin: 输入参数xmin的最小值
    :param xmax: 输入参数xmax的最大值
    :return: 输出参数y：归一化到区间[omin,omax]的数据
    """
    xmax = np.max(x) if xmax is None else xmax  # %计算最大值
    xmin = np.min(x) if xmin is None else xmin  # %计算最小值
    y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
    return y


def cv_image_normalize(image, min_val=0.0, max_val=1.0):
    """
    :param image:
    :param min_val:
    :param max_val:
    :param norm_type:
    :param dtype:
    :param mask:
    :return:
    """
    dtype = cv2.CV_32F
    norm_type = cv2.NORM_MINMAX
    out = np.zeros(shape=image.shape, dtype=np.float32)
    cv2.normalize(image, out, alpha=min_val, beta=max_val, norm_type=norm_type, dtype=dtype)
    return out


def get_prewhiten_images(images_list, norm=False):
    """
    批量白化图片处理
    :param images_list:
    :param norm:
    :return:
    """
    out_images = []
    for image in images_list:
        if norm: image = image_normalize(image)
        image = get_prewhiten_image(image)
        out_images.append(image)
    return out_images


def read_image(filename, size=None, norm=False, use_rgb=False):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param size: (W,H)
    :param norm:是否归一化到[0.,1.0]
    :param use_rgb 输出格式：RGB(True) or BGR(False)
    :return: 返回的图片数据
    """
    image = cv2.imread(filename)
    # image = cv2.imread(filename, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    if size: image = resize_image(image, size=size)
    if norm: image = image_normalize(image)
    return image


def read_image_ch(filename, size=None, norm=False, use_rgb=False):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param size: (W,H)
    :param norm:是否归一化到[0.,1.0]
    :param use_rgb 输出格式：RGB or BGR
    :return: 返回的图片数据
    """
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径
    if image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    if size: image = resize_image(image, size=size)
    if norm: image = image_normalize(image)
    return image


def read_image_pil(filename, size=None, norm=False, use_rgb=False):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param size: (W,H)
    :param norm:是否归一化到[0.,1.0]
    :param use_rgb 输出格式：RGB or BGR
    :return: 返回的图片数据
    """
    image = Image.open(filename)  # is RGB
    image = np.asarray(image)
    if len(image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if not use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    if size: image = resize_image(image, size=size)
    if norm: image = image_normalize(image)
    return image


def requests_url(url, timeout=None):
    """
    读取网络数据流
    :param url:
    :return:
    """
    stream = None
    try:
        res = requests.get(url, timeout=timeout)
        if res.status_code == 200:
            stream = res.content
    except Exception as e:
        print(e)
    return stream


def read_images_url(url: str, size=None, norm=False, use_rgb=False, timeout=5):
    """
    根据url或者图片路径，读取图片
    :param url:
    :param size:
    :param norm:
    :param use_rgb:
    :return:
    """
    image = None
    # if re.match(r'^https?:/{2}\w.+$', url):
    if url.startswith("http"):
        stream = requests_url(url, timeout=timeout)
        if stream is not None:
            content = np.asarray(bytearray(stream), dtype="uint8")
            image = cv2.imdecode(content, cv2.IMREAD_COLOR)
            # pil_image = PIL.Image.open(BytesIO(stream))
            # image=np.asarray(pil_image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(url)
    if image is None:
        print("Warning: no image:{}".format(url))
        return None
    if len(image.shape) == 2:  # 若是灰度图则转为三通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    if size: image = resize_image(image, size=size)
    if norm: image = image_normalize(image)
    return image


def read_image_batch(image_list):
    """
    批量读取图片
    :param image_list:
    :return:
    """
    image_batch = []
    out_image_list = []
    for image_path in image_list:
        image = read_images_url(image_path)
        if image is None:
            print("no image:{}".format(image_path))
            continue
        image_batch.append(image)
        out_image_list.append(image_path)
    return image_batch, out_image_list


def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, norm=False, use_rgb=True):
    """
    快速读取图片的方法
    :param filename: 图片路径
    :param orig_rect:原始图片的感兴趣区域rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param norm: 是否归一化
    :param use_rgb 输出格式：RGB or BGR
    :return: 返回感兴趣区域ROI
    """
    # 当采用IMREAD_REDUCED模式时，对应rect也需要缩放
    scale = 1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale = 1 / 2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale = 1 / 4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale = 1 / 8
    rect = np.array(orig_rect) * scale
    rect = rect.astype(int).tolist()
    image = cv2.imread(filename, flags=ImreadModes)
    if image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    if norm: image = image_normalize(image)
    roi_image = get_rect_image(image, rect)
    return roi_image


def resize_scale_image(image, size: int, use_length=True, interpolation=cv2.INTER_LINEAR):
    """
    按照长/短边进行等比例缩放
    :param image:
    :param size: 目标长度
    :param use_length: True长边对齐缩放，False: 短边对齐缩放
    :return:
    """
    height, width = image.shape[:2]
    r = [size / width, size / height]
    r = min(r) if use_length else max(r)
    dimage = cv2.resize(image, dsize=(int(width * r), int(height * r)), interpolation=interpolation)
    return dimage


def resize_image_padding(image, size, color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    """
    为保证图像无形变，需要进行等比例缩放和填充
    :param image:
    :param color: 短边进行填充的color value
    :return:
    """
    if not size: return image
    h, w = image.shape[:2]
    if w / h > size[0] / size[1] > 0:
        dsize = (size[0], None)
    else:
        dsize = (None, size[1])
    image = resize_image(image, size=dsize, interpolation=interpolation)
    image = center_crop_padding(image, crop_size=size, color=color)
    return image


def resize_image_clip(image, clip_max=1920, interpolation=cv2.INTER_LINEAR):
    """
    限制图像最大分辨率
    :param image:
    :param clip_max:
    :param interpolation:
    :return:
    """
    height, width = image.shape[:2]
    if height > width:
        width, height = None, min(height, clip_max)
    else:
        width, height = min(width, clip_max), None
    image = resize_image(image, size=(width, height), interpolation=interpolation)
    return image


def resize_image(image, size: Tuple, interpolation=cv2.INTER_LINEAR):
    """
    tf.image.resize_images(images,size),images=[batch, height, width, channels],size=(new_height, new_width)
    cv2.resize(image, dsize=(width, height)),与image.shape相反
    images[50,10]与image.shape的原理相同，它表示的是image=(y=50,x=10)
    :param image:
    :param size: (width,height)
    :return:
    """
    if not size: return image
    size = (size, size) if len(size) == 1 else size
    dw, dh = size
    sh, sw = image.shape[:2]
    if (dh is None) and (dw is None):  # 错误写法：resize_height and resize_width is None
        return image
    if dh is None:
        dh = int(sh * dw / sw)
    elif dw is None:
        dw = int(sw * dh / sh)
    image = cv2.resize(image, dsize=(int(dw), int(dh)), interpolation=interpolation)
    return image


def image_boxes_resize_padding(image, input_size, boxes=None, points=None, color=(0, 0, 0)):
    """
    等比例图像resize,保持原始图像内容比，避免失真,短边会0填充
    input_size = [300, 300]
    image_path = "test.jpg"
    src_boxes = [[8.20251, 1, 242.2412, 699.2236],
                 [201.14865, 204.18265, 468.605, 696.36163]]
    src_boxes = np.asarray(src_boxes)
    image = read_image(image_path)
    image1, boxes1 = image_boxes_resize_padding(image, input_size, src_boxes)
    image1 = show_image_boxes("Det", image1, boxes1, color=(255, 0, 0), delay=3)
    boxes = image_boxes_resize_padding_inverse(image.shape, input_size, boxes1)
    show_image_boxes("image", image, boxes)
    :param size:
    """
    height, width = image.shape[:2]
    scale = min([input_size[0] / width, input_size[1] / height])
    new_size = [int(width * scale), int(height * scale)]
    pad_w = input_size[0] - new_size[0]
    pad_h = input_size[1] - new_size[1]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    out = cv2.resize(image, (new_size[0], new_size[1]))
    out = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not boxes is None and len(boxes) > 0:
        boxes[:] = boxes[:] * scale
        boxes[:] = boxes[:] + [left, top, left, top]
    if not points is None and len(points) > 0:
        points[:] = points[:] * scale
        points[:] = points[:] + [left, top]
    return out


def image_boxes_resize_padding_inverse(image_size, input_size, boxes=None, points=None):
    """
    image_boxes_resize_padding的逆过程
    :param image_size: 原始图像的大小(W,H)
    :param input_size: 模型输入大小(W,H)
    :param boxes: 输入/输出检测框(n,4)
    :param points: 输入/输出关键点(n,4)
    :return:
    """
    width, height = image_size
    scale = min([input_size[0] / width, input_size[1] / height])
    new_size = [int(width * scale), int(height * scale)]
    pad_w = input_size[0] - new_size[0]
    pad_h = input_size[1] - new_size[1]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    if not boxes is None and len(boxes) > 0:
        boxes[:] = boxes[:] - [left, top, left, top]
        boxes[:] = boxes[:] / scale
    if not points is None and len(points) > 0:
        points[:] = points[:] - [left, top]
        points[:] = points[:] / scale
    return boxes


def resize_image_boxes(image, size, boxes=None):
    """
    :param image:
    :param size: (W,H)
    :param boxes:
    :return:
    """
    resize_width, resize_height = size
    height, width, _ = image.shape
    if (resize_height is None) and (resize_width is None):  # 错误写法：resize_height and resize_width is None
        return image, boxes
    if resize_height is None:
        scale = [resize_width / width, resize_width / width]
        resize_height = int(height * resize_width / width)
    elif resize_width is None:
        scale = [resize_height / height, resize_height / height]
        resize_width = int(width * resize_height / height)
    else:
        scale = [resize_width / width, resize_height / height]
    boxes = scale * 2 * boxes
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image, boxes


def scale_image(image, scale):
    """
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    """
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])
    return image


def get_rect_image(image, rect):
    """
    :param image:
    :param rect: [x,y,w,h]
    :return:
    """
    shape = image.shape  # h,w
    height = shape[0]
    width = shape[1]
    image_rect = (0, 0, width, height)
    rect = get_rect_intersection(rect, image_rect)
    rect = [int(i) for i in rect]
    x, y, w, h = rect
    cut_image = image[y:(y + h), x:(x + w)]
    return cut_image


def get_rects_image(image, rects_list, size):
    """
    获得裁剪区域
    :param image:
    :param rects_list:
    :param size:
    :return:
    """
    rect_images = []
    for rect in rects_list:
        roi = get_rect_image(image, rect)
        roi = resize_image(roi, size=size)
        rect_images.append(roi)
    return rect_images


def get_boxes_image(image, boxes, size):
    """
    获得裁剪区域
    :param image:
    :param boxes:
    :param size:
    :return:
    """
    rects = boxes2rects(boxes)
    crops = get_rects_image(image, rects, size=size)
    return crops


get_bboxes_image = get_boxes_image


def boxes2rects(boxes):
    """
    将boxes=[x1,y1,x2,y2] 转为rect=[x1,y1,w,h]
    :param boxes:
    :return:
    """
    rects_list = []
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = [x1, y1, (x2 - x1), (y2 - y1)]
        rects_list.append(rect)
    return rects_list


bboxes2rects = boxes2rects


def rects2boxes(rects):
    """
    将rect=[x1,y1,w,h]转为boxes=[x1,y1,x2,y2]
    :param rects:
    :return:
    """
    boxes = []
    for rect in rects:
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h
        b = (x1, y1, x2, y2)
        boxes.append(b)
    return boxes


rects2bboxes = rects2boxes


def boxes2center(boxes):
    """
    center = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    将boxes=[x1,y1,x2,y2] 转为center_list=[cx,cy,w,h]
    :param boxes:
    :return:
    """
    center_list = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
        center_list.append(center)
    return center_list


bboxes2center = boxes2center


def center2boxes(center):
    """
    将center_list=[cx,cy,w,h] 转为boxes=[x1,y1,x2,y2]
    :param center
    :return:
    """
    boxes = []
    for c in center:
        cx, cy, w, h = c
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        boxes.append(box)
    return boxes


center2bboxes = center2boxes


def center2rects(center):
    """
    将center_list=[cx,cy,w,h] 转为rect=[x,y,w,h]
    :param center:
    :return:
    """
    rect_list = []
    for c in center:
        cx, cy, w, h = c
        rect = [cx - w / 2, cy - h / 2, w, h]
        rect_list.append(rect)
    return rect_list


def scale_rect(orig_rect, orig_shape, dest_shape):
    """
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    """
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect


def scale_box(orig_box, orig_shape, dest_shape):
    """
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_box: 原始图像的box=[xmin,ymin,xmax,ymax]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    """
    r = (dest_shape[1] / orig_shape[1], dest_shape[0] / orig_shape[0])
    dest_box = [int(orig_box[0] * r[0]), int(orig_box[1] * r[1]),
                int(orig_box[2] * r[0]), int(orig_box[3] * r[1])]
    return dest_box


def get_rect_intersection(rec1, rec2):
    """
    计算两个rect的交集坐标
    :param rec1:
    :param rec2:
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = rects2boxes([rec1])[0]
    xmin2, ymin2, xmax2, ymax2 = rects2boxes([rec2])[0]
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return (x1, y1, w, h)


def get_box_intersection(box1, box2):
    """
    计算两个boxes的交集坐标
    :param rec1:
    :param rec2:
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    return (x1, y1, x2, y2)


def draw_image_rects(image, rects, color=(0, 0, 255), thickness=-1):
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=-1.0)
    for rect in rects:
        x, y, w, h = rect
        point1 = (int(x), int(y))
        point2 = (int(x + w), int(y + h))
        cv2.rectangle(image, point1, point2, color, thickness=thickness)
    return image


def draw_image_boxes(image, boxes, color=(0, 0, 255), thickness=-1):
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=-1.0)
    if isinstance(color[0], numbers.Number): color = [color] * len(boxes)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, color[i], thickness=thickness)
    return image


def show_image_rects(title, image, rects, color=(0, 0, 255), delay=0, use_rgb=False):
    """
    :param title:
    :param image:
    :param rects:[[ x, y, w, h],[ x, y, w, h]]
    :return:
    """
    image = draw_image_rects(image, rects, color)
    cv_show_image(title, image, delay=delay, use_rgb=use_rgb)
    return image


def show_image_boxes(title, image, boxes, color=(0, 0, 255), delay=0, use_rgb=False):
    """
    :param title:
    :param image:
    :param boxes:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    """
    image = draw_image_boxes(image, boxes, color)
    cv_show_image(title, image, delay=delay, use_rgb=use_rgb)
    return image


def draw_image_bboxes_text(image, boxes, boxes_name, color=(), thickness=-1, fontScale=-1.0,
                           drawType="custom", top=True):
    """
    已经废弃，使用draw_image_boxes_texts代替
    :param boxes_name:
    :param bgr_image: bgr image
    :param color: BGR color:[B,G,R]
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    """
    if isinstance(boxes_name, np.ndarray):
        boxes_name = boxes_name.reshape(-1).tolist()
    for i, (name, box) in enumerate(zip(boxes_name, boxes)):
        box = [int(b) for b in box]
        c = color if color else color_table[i + 1]
        draw_image_box_text(image, box, c, name, thickness, fontScale, drawType=drawType, top=top)
    return image


def draw_image_boxes_texts(image, boxes, texts, color=(), thickness=-1, fontScale=-1.0, drawType="custom", top=True):
    """
    :param image:
    :param boxes:
    :param texts:
    :param color:
    :param thickness:
    :param fontScale:
    :param drawType:
    :param top:
    :return:
    """
    for i, (box, text) in enumerate(zip(boxes, texts)):
        box = [int(b) for b in box]
        c = color if color else color_table[i + 1]
        draw_image_box_text(image, box, c, text, thickness, fontScale, drawType=drawType, top=top)
    return image


def draw_image_bboxes_labels_text(image, boxes, labels, boxes_name=None, color=None, thickness=2, fontScale=0.8,
                                  drawType="custom", top=True):
    """已废弃，使用draw_image_boxes_labels_texts代替"""
    if isinstance(labels, np.ndarray):
        labels = labels.reshape(-1).tolist()
    boxes_name = boxes_name if boxes_name else labels
    for label, box, name in zip(labels, boxes, boxes_name):
        box = [int(b) for b in box]
        color_ = color if color else color_map[int(label) + 1]
        image = draw_image_box_text(image, box, color_, str(name), thickness, fontScale, drawType, top)
    return image


def draw_image_boxes_labels_texts(image, boxes, labels, texts=None, color=None, thickness=2, fontScale=0.8,
                                  drawType="custom", top=True):
    if isinstance(labels, np.ndarray):
        labels = labels.reshape(-1).tolist()
    texts = texts if texts else labels
    for label, box, name in zip(labels, boxes, texts):
        box = [int(b) for b in box]
        color_ = color if color else color_map[int(label) + 1]
        image = draw_image_box_text(image, box, color_, str(name), thickness, fontScale, drawType, top)
    return image


def draw_image_rects_labels_text(image, rects, labels, boxes_name=None, color=None, thickness=2, fontScale=0.8,
                                 drawType="custom", top=True):
    """已废弃，使用draw_image_rects_labels_texts代替"""
    boxes = rects2boxes(rects)
    image = draw_image_boxes_labels_texts(image, boxes, labels, texts=boxes_name, color=color, thickness=thickness,
                                          fontScale=fontScale, drawType=drawType, top=top)
    return image


def draw_image_rects_labels_texts(image, rects, labels, texts=None, color=None, thickness=2, fontScale=0.8,
                                  drawType="custom", top=True):
    boxes = rects2boxes(rects)
    image = draw_image_boxes_labels_texts(image, boxes, labels, texts=texts, color=color, thickness=thickness,
                                          fontScale=fontScale, drawType=drawType, top=top)
    return image


def show_image_boxes_texts(title, image, boxes, texts, color=None, thickness=-1, fontScale=-1.0,
                           drawType="custom", delay=0, top=True):
    """
    :param texts:
    :param bgr_image: bgr image
    :param color: BGR color:[B,G,R]
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    """
    image = draw_image_boxes_texts(image, boxes, texts=texts, color=color, thickness=thickness,
                                   fontScale=fontScale, drawType=drawType, top=top)
    cv_show_image(title, image, delay=delay)
    return image


def draw_image_rects_text(image, rects, texts, color=None, thickness=-1, fontScale=-1.0,
                          drawType="custom", top=True):
    """已经废弃，使用draw_image_rects_texts代替"""
    boxes = rects2boxes(rects)
    image = draw_image_boxes_texts(image, boxes, texts=texts, color=color, thickness=thickness,
                                   fontScale=fontScale, drawType=drawType, top=top)
    return image


def draw_image_rects_texts(image, rects, texts, color=None, thickness=-1, fontScale=-1.0,
                           drawType="custom", top=True):
    boxes = rects2boxes(rects)
    image = draw_image_boxes_texts(image, boxes, texts=texts, color=color, thickness=thickness,
                                   fontScale=fontScale, drawType=drawType, top=top)
    return image


def show_image_rects_text(title, image, rects, rects_name, color=None, thickness=-1, fontScale=-1.0, drawType="custom",
                          delay=0, top=True):
    """
    已经废弃，使用show_image_rects_texts代替
    :param rects_name:
    :param bgr_image: bgr image
    :param rects: [[x1,y1,w,h],[x1,y1,w,h]]
    :return:
    """
    boxes = rects2boxes(rects)
    image = show_image_boxes_texts(title, image, boxes, texts=rects_name, color=color, thickness=thickness,
                                   fontScale=fontScale, drawType=drawType, delay=delay, top=top)
    return image


def show_image_rects_texts(title, image, rects, texts, color=None, thickness=-1, fontScale=-1.0, drawType="custom",
                           delay=0, top=True):
    """
    :param texts:
    :param image: image
    :param rects: [[x1,y1,w,h],[x1,y1,w,h]]
    :return:
    """
    boxes = rects2boxes(rects)
    image = show_image_boxes_texts(title, image, boxes, texts=texts, color=color, thickness=thickness,
                                   fontScale=fontScale, drawType=drawType, delay=delay, top=top)
    return image


def draw_image_bboxes_labels(image, boxes, labels, class_name=None, color=None,
                             thickness=-1, fontScale=-1.0, drawType="custom"):
    """
    :param image:
    :param boxes:  [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :param labels:
    :return:
    """
    if isinstance(labels, np.ndarray): labels = labels.astype(np.int32).reshape(-1).tolist()
    for label, box in zip(labels, boxes):
        box = [int(b) for b in box]
        name = label
        color_ = color
        if isinstance(name, numbers.Number) and class_name:
            color_ = color_table[int(name) + 1]
            name = class_name[int(name)]
        if not color_: color_ = color_table[1]
        image = draw_image_box_text(image, box, color_, str(name), thickness=thickness,
                                    fontScale=fontScale, drawType=drawType)
    return image


draw_image_boxes_labels = draw_image_bboxes_labels


def draw_image_rects_labels(image, rects, labels, class_name=None, color=None, thickness=-1, fontScale=-1.0):
    """
    :param image:
    :param rects:
    :param labels:
    :return:
    """
    boxes = rects2boxes(rects)
    image = draw_image_boxes_labels(image, boxes, labels, class_name=class_name, color=color,
                                    thickness=thickness, fontScale=fontScale)
    return image


def draw_image_detection_rects(image, rects, probs, labels, class_name=None, thickness=-1, fontScale=-1.0,
                               drawType="custom"):
    """
    :param image:
    :param rects:
    :param probs:
    :param labels:
    :param class_name:
    :param thickness:
    :param fontScale:
    :param drawType:
    :return:
    """
    boxes = rects2boxes(rects)
    image = draw_image_detection_boxes(image, boxes, probs, labels, class_name,
                                       thickness=thickness, fontScale=fontScale, drawType=drawType)
    return image


def draw_image_detection_boxes(image, boxes, probs, labels, class_name=None, thickness=-1, fontScale=-1.0,
                               drawType="custom"):
    """
    :param image:
    :param boxes:
    :param probs:
    :param labels:
    :param class_name:
    :param thickness:
    :param fontScale:
    :param drawType:
    :return:
    """
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    labels = labels if isinstance(labels, list) else np.asarray(labels, dtype=np.int32).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    for label, box, prob in zip(labels, boxes, probs):
        color = color_table[1] if isinstance(label, str) else color_table[int(label) + 1]
        box = [int(b) for b in box]
        if class_name:
            label = class_name[int(label)]
        boxes_name = "{}:{:3.2f}".format(label, prob)
        draw_image_box_text(image, box, color, boxes_name, thickness=thickness, fontScale=fontScale, drawType=drawType)
    return image


draw_image_detection_bboxes = draw_image_detection_boxes


def get_linesize(length, thickness=-1, fontScale=-1.0):
    """
    自动计算绘图的大小thickness，fontScale
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    """
    if fontScale <= 0: fontScale = max(2.0 * length / 850.0, 0.6)
    if thickness <= 0: thickness = int(2.0 * fontScale + 1)
    return thickness, fontScale


def draw_dt_gt_dets(image, dt_boxes, dt_label, gt_boxes, gt_label, vis_diff=False):
    """
    显示ground true和Detection bbox
    :param image:
    :param dt_boxes:
    :param dt_label:
    :param gt_boxes:
    :param gt_label:
    :param vis_diff: 是否显示差异，
                vis_diff=True :使用不同颜色标记gt_label和dt_label的差异
                               要求len(gt_label) == len(dt_label)
                vis_diff=False:使用不同颜色显示dt_boxes和gt_boxes
                                dt_boxes和gt_boxes的长度任意
    :return:
    """

    if vis_diff:
        assert len(gt_label) == len(dt_label)
        assert len(gt_boxes) == len(dt_boxes)
        for i in range(len(gt_label)):
            if dt_label[i] == gt_label[i]:
                image = draw_image_boxes_texts(image, [gt_boxes[i]], [gt_label[i]], color=(0, 255, 0))
                image = draw_image_boxes_texts(image, [dt_boxes[i]], [dt_label[i]], color=(0, 255, 0))
            else:
                image = draw_image_boxes_texts(image, [gt_boxes[i]], [gt_label[i]], color=(0, 255, 0))
                image = draw_image_boxes_texts(image, [dt_boxes[i]], [dt_label[i]], color=(255, 0, 0))
    else:
        image = draw_image_boxes_texts(image, gt_boxes, gt_label, color=(0, 255, 0))
        image = draw_image_boxes_texts(image, dt_boxes, dt_label, color=(255, 0, 0))
    return image


def draw_landmark(image, landmarks, radius=2, fontScale=1.0, color=(0, 0, 255), vis_id=False):
    """
    :param image:
    :param landmarks:
    :param radius:
    :param thickness:
    :param fontScale:
    :param color:
    :param vis_id:
    :return:
    """
    image = copy.copy(image)
    for lm in landmarks:
        for i, landmark in enumerate(lm):
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)
            if vis_id:
                image = draw_points_text(image, [point], texts=[str(i)], color=color, thickness=radius,
                                         fontScale=fontScale, drawType="simple")
    return image


def show_landmark_boxes(title, image, landmarks, boxes):
    """
    显示landmark和boxex
    :param title:
    :param image:
    :param landmarks: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    """
    image = draw_landmark(image, landmarks)
    image = show_image_boxes(title, image, boxes)
    return image


def show_landmark(title, image, landmarks, vis_id=False, delay=0):
    """
    显示landmark和boxex
    :param title:
    :param image:
    :param landmarks: [[x1, y1], [x2, y2]]
    :return:
    """
    image = draw_landmark(image, landmarks, vis_id=vis_id)
    cv_show_image(title, image, delay=delay)
    return image


def draw_points_text(image, points, texts=None, color=(255, 0, 0), thickness=-1, fontScale=-1.0, drawType="simple"):
    """
    :param image:
    :param points:
    :param texts:
    :param color:
    :param drawType: custom or simple
    :return:
    """
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    if texts is None: texts = [""] * len(points)
    if isinstance(color[0], numbers.Number): color = [color] * len(points)
    for index, (point, text) in enumerate(zip(points, texts)):
        if not check_point(point): continue
        point = (int(point[0]), int(point[1]))
        c = color[index]
        text = str(text)
        cv2.circle(image, point, thickness * 3, tuple(c), -1, lineType=cv2.LINE_AA)
        if text: draw_text(image, point, text, color=c, fontScale=fontScale, thickness=thickness, drawType=drawType)
    return image


draw_points_texts = draw_points_text


def draw_texts(image, points, texts, color=(255, 0, 0), fontScale=-1.0, thickness=-1, drawType="simple"):
    for point, text in zip(points, texts):
        image = draw_text(image, point, text, color=color, fontScale=fontScale, thickness=thickness, drawType=drawType)
    return image


def draw_text(image, point, text, color=(255, 0, 0), fontScale=-1.0, thickness=-1, drawType="custom"):
    """
    :param image:
    :param point:
    :param text:
    :param drawType: custom or simple
    :return:
    """
    point = (int(point[0]), int(point[1]))
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom" or drawType == "en":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(image, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), color=color, thickness=-1)
        # draw score value
        cv2.putText(image, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale, (255, 255, 255),
                    thickness, 2)
    elif drawType == "simple":
        cv2.putText(image, str(text), (point[0], point[1]), fontFace, fontScale, color=color, thickness=thickness)
    if drawType == "chinese" or drawType == "ch":
        cv2_putText(image, str(text), point, color=color, fontScale=fontScale, thickness=thickness)
    return image


def draw_image_box_text(image, bbox, color, text, thickness=2, fontScale=0.8, drawType="custom", top=True):
    """
    :param image:
    :param bbox:
    :param color:
    :param text:
    :param drawType:
    :param top:
    :return:
    """
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    text_loc = (bbox[0], bbox[1]) if top else (bbox[0], bbox[3])
    if not text: drawType = "simple"
    if drawType == "chinese" or drawType == "ch":
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        cv2_putText(image, str(text), text_loc, color=color, fontScale=fontScale, thickness=thickness)
    elif drawType == "simple" or drawType == "en":
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness, 8, 0)
        cv2.putText(image, str(text), text_loc, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
    elif drawType == "custom":
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        text_size, baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        if top:
            # text_loc = (bbox[0], bbox[1] + text_size[1])  # 在左上角下方绘制文字
            text_loc = (bbox[0], bbox[1] - baseline // 2)  # 在左上角上方绘制文字
        else:
            text_loc = (bbox[0], bbox[3])
        bg1 = (text_loc[0], text_loc[1] - text_size[1])
        bg2 = (text_loc[0] + text_size[0], text_loc[1] + baseline // 2)  # 底纹等于字体的宽度
        # bg2 = (max(bbox[2], text_loc[0] + text_size[0]), text_loc[1] + baseline // 2) # 底纹等于box的宽度
        cv2.rectangle(image, bg1, bg2, color, thickness)  # 先绘制框，再填充
        cv2.rectangle(image, bg1, bg2, color, -1)
        # draw score value
        cv2.putText(image, str(text), (text_loc[0], text_loc[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    (255, 255, 255), thickness)
    return image


def draw_text_line(image, point, text_line: str, bg_color=(255, 0, 0), thickness=-1, fontScale=-1.0, drawType="custom"):
    """
    :param image:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    """
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = (point[0], point[1] + (text_size[1] + 2 + baseline) * i)
            image = draw_text(image, draw_point, text, bg_color, thickness=thickness, drawType=drawType)
    return image


def draw_text_pil(image, point, text, size=10, color_color=(255, 0, 0)):
    """
    支持显示中文
    :param image:
    :param point:
    :param text:
    :param size:
    :param color_color:
    :return:
    """
    pilimg = Image.fromarray(image)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    # 参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：
    font = ImageFont.truetype(os.path.join(root, "font_style/simhei.ttf"), size, encoding="utf-8")
    draw.text(point, text, color_color, font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式
    return image


def cv2_putText(img, text, point, fontFace=None, fontScale=0.8, color=(255, 0, 0), thickness=None):
    # cv2.putText(img, str(text), point, fontFace, fontScale, color=color, thickness=thickness)
    pilimg = Image.fromarray(img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    size = int(fontScale * 10 * thickness)
    point = (point[0], point[1] - int(size * 1.3))
    font = get_font_type(size=size)
    draw.text(point, text, color, font)
    img[:] = np.asarray(pilimg)


def draw_key_point_in_image(image, key_points, pointline=[], boxes=[], colors=None, vis_id=False, thickness=2):
    """
    :param image:
    :param key_points: list(ndarray(19,2)) or ndarray(n_person,19,2)
    :param pointline: `auto`->pointline = circle_line(len(points), iscircle=True)
    :param boxes: 目标框
    :param colors: 每个点的颜色
    :return:
    """
    nums = max(len(key_points), len(boxes))
    for p in range(nums):
        color = color_table[p + 1] if not colors else colors
        if len(key_points) > 0:
            points = key_points[p]
            if points is None or len(points) == 0: continue
            text = list(range(len(points))) if vis_id else [""] * len(points)
            image = draw_image_points_lines(image, points, pointline, circle_color=color, line_color=color,
                                            texts=text, thickness=thickness)
        if len(boxes) > 0:
            image = draw_image_boxes(image, boxes=[boxes[p]], color=color, thickness=thickness)
    return image


def draw_key_point_arrowed_in_image(image, key_points, pointline=[], color=None):
    """
    :param key_points: list(ndarray(19,2)) or ndarray(n_person,19,2)
    :param image:
    :param pointline:[[start-->end]]
    :return:
    """
    for person_id, points in enumerate(key_points):
        if points is None or len(points) == 0: continue
        image = draw_image_points_arrowed_lines(image, points, pointline)
    return image


def draw_image_points_lines(image,
                            points,
                            pointline=[],
                            texts=None,
                            fontScale=0.8,
                            circle_color=(255, 0, 0),
                            line_color=(0, 0, 255),
                            thickness=2):
    """
    在图像中画点和连接线
    :param image:
    :param points: 点列表
    :param pointline: `auto`->pointline = circle_line(len(points), iscircle=True)
    :param color:
    :param texts:
    :param drawType: simple or custom
    :param check:
    :return:
    """
    points = np.asarray(points, dtype=np.int32)
    if texts is None:
        texts = list(range(len(points)))
    draw_image_lines(image, points, pointline, color=line_color, thickness=thickness)
    thickness_ = max(int(thickness * 1), 1)
    image = draw_points_text(image, points,
                             texts=texts,
                             color=circle_color,
                             fontScale=fontScale,
                             thickness=thickness_,
                             drawType="simple")
    return image


def draw_image_points_arrowed_lines(image, points,
                                    pointline=[],
                                    texts=None,
                                    drawType="simple",
                                    reverse=False):
    """
    在图像中画点箭头线
    :param image:
    :param points: 点列表
    :param pointline: `auto`->pointline = circle_line(len(points), iscircle=True)
    :param color:
    :param texts:
    :param drawType: simple or custom
    :param check:
    :return:
    """
    points = np.asarray(points, dtype=np.int32)
    thickness = 2
    if texts is None:
        texts = list(range(len(points)))
    image = draw_points_text(image, points, texts=texts, thickness=thickness, drawType=drawType)
    draw_image_arrowed_lines(image, points, pointline, thickness=thickness, reverse=reverse)
    return image


def draw_image_lines(image, points, pointline=[], color=(0, 0, 255), thickness=2):
    # points = np.asarray(points, dtype=np.int32)
    if pointline == "auto" or pointline == []:
        pointline = circle_line(len(points), iscircle=True)
    if isinstance(color[0], numbers.Number): color = [color] * len(points)
    for index in pointline:
        point1 = tuple(points[index[0]])
        point2 = tuple(points[index[1]])
        if (not check_point(point1)) or (not check_point(point2)):
            continue
        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]), int(point2[1]))
        cv2.line(image, point1, point2, color[index[1]], thickness)  # 绿色，3个像素宽度
    return image


def draw_image_arrowed_lines(image,
                             points,
                             pointline=[],
                             color=(0, 0, 255),
                             thickness=2,
                             reverse=False):
    points = np.asarray(points, dtype=np.int32)
    if pointline == "auto":
        pointline = circle_line(len(points), iscircle=True)
    for point_index in pointline:
        point1 = tuple(points[point_index[0]])
        point2 = tuple(points[point_index[1]])
        if (not check_point(point1)) or (not check_point(point2)):
            continue
        if reverse:
            cv2.arrowedLine(image, point1, point2, color, thickness=thickness)
        else:
            cv2.arrowedLine(image, point2, point1, color, thickness=thickness)
    return image


def draw_image_polylines(image, points, color=(0, 0, 255)):
    """
    # points是三维坐标，分别表示(多边形个数，多边形坐标点x,多边形坐标点y)=(num_polylines,num_point,2)
    points = np.asarray([[[100, 100], [200, 100], [400, 200]],
                         [[500, 600], [300, 400], [500, 700]]]
                         )
    cv2.polylines(image,points,True,(255,0,0))  #画任意多边形
    参数2 pts：多边形的顶点坐标(按顺序)，points是三维坐标，分别表示(多边形个数，多边形坐标点x,多边形坐标点y)
    参数3 isClosed：必选参数。用于设置绘制的折线是否关闭，若设置为True，则从折线的最后一个顶点到其第一个顶点会自动绘制一条线进行闭合。
    参数4 color：必选参数。用于设置多边形的颜色
    参数5 lineType：可选参数。用于设置线段的类型，可选8（8邻接连接线-默认）、4（4邻接连接线）和cv2.LINE_AA 为抗锯齿
    """
    points = np.asarray(points, dtype=np.int32)
    image = cv2.polylines(image, points, isClosed=True, color=color, thickness=2)  # 画任意多边形
    return image


def draw_image_fillPoly(image, points, color=(0, 0, 255)):
    """
    # points是三维坐标，分别表示(多边形个数，多边形坐标点x,多边形坐标点y)=(num_polylines,num_point,2)
    points = np.asarray([[[100, 100], [200, 100], [400, 200]],
                         [[500, 600], [300, 400], [500, 700]]]
                         )
    cv2.polylines(image,points,True,(255,0,0))  #画任意多边形
    参数2 pts：多边形的顶点坐标(按顺序)，points是三维坐标，分别表示(多边形个数，多边形坐标点x,多边形坐标点y)
    参数3 isClosed：必选参数。用于设置绘制的折线是否关闭，若设置为True，则从折线的最后一个顶点到其第一个顶点会自动绘制一条线进行闭合。
    参数4 color：必选参数。用于设置多边形的颜色
    参数5 lineType：可选参数。用于设置线段的类型，可选8（8邻接连接线-默认）、4（4邻接连接线）和cv2.LINE_AA 为抗锯齿
    """
    points = np.asarray(points, dtype=np.int32)
    image = cv2.fillPoly(image, points, color=color)
    return image


def circle_line(num_point, iscircle=True):
    """
    产生连接线的点,用于绘制连接线
    points_line=circle_line(len(points),iscircle=True)
    >> [(0, 1), (1, 2), (2, 0)]
    :param num_point:
    :param iscircle: 首尾是否相连
    :return:
    """
    start = 0
    end = num_point - 1
    points_line = []
    for i in range(start, end + 1):
        if i == end and iscircle:
            points_line.append([end, start])
        elif i != end:
            points_line.append([i, i + 1])
    return points_line


def cv_paste_image(im, mask, start_point=(0, 0)):
    """
    :param im:
    :param start_point:
    :param mask:
    :return:
    """
    xim, ymin = start_point
    shape = mask.shape  # h, w, d
    im[ymin:(ymin + shape[0]), xim:(xim + shape[1])] = mask
    return im


def pil_paste_image(im, mask, start_point=(0, 0)):
    """
    :param im:
    :param mask:
    :param start_point:
    :return:
    """
    out = Image.fromarray(im)
    mask = Image.fromarray(mask)
    out.paste(mask, start_point.copy())  # fix a bug
    return np.asarray(out)


def image_rotation(image, angle, center=None, scale=1.0, borderValue=(0, 0, 0)):
    """
    图像旋转
    :param image:
    :param angle:
    :param center:
    :param scale:
    :param borderValue:
    :return:
    """
    h, w = image.shape[:2]
    if not center:
        # center = (w // 2, h // 2)
        center = (w / 2., h / 2.)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, mat, dsize=(w, h), borderMode=cv2.BORDER_REFLECT_101, borderValue=borderValue)
    return rotated


def save_image(image_file, image, uint8=False, use_rgb=False):
    """
    保存图片
    :param image_file: 保存图片文件
    :param image: BGR Image
    :param uint8: 乘以255并转换为INT8类型
    :param use_rgb: 是否将输入image(BGR)转换为RGB
    :return:
    """
    save_dir = os.path.dirname(image_file)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if uint8: image = np.asarray(image * 255, dtype=np.uint8)
    if use_rgb: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_file, image)


def nms_boxes_cv2(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, score_threshold=0.5, nms_threshold=0.45,
                  eta=None, top_k=None, use_class=True):
    """
    NMS
    fix a bug: cv2.dnn.NMSBoxe bboxes, scores params must be list and float data,can not be float32 or int
    :param boxes: List or np.ndarray,shape is（nums-boxes,4）is (x,y,w,h)
    :param scores: List or np.ndarray,shape is（nums-boxes,）
    :param labels: List or np.ndarray,shape is（nums-boxes,）
    :param image_size: (width,height)
    :param score_threshold:
    :param nms_threshold:
    :return:
    """
    if use_class:
        # TODO opencv4.7.0版本中增加了NMSBoxesBatched函数，可分类做nms
        index = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), labels.tolist(),
                                        score_threshold=score_threshold, nms_threshold=nms_threshold,
                                        eta=eta, top_k=top_k)
    else:
        # TODO NMSBoxes没有区分类别
        index = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=score_threshold,
                                 nms_threshold=nms_threshold, eta=eta, top_k=top_k)
    return index


def file2base64(file):
    image_base64 = base64.b64encode(open(file, 'rb').read()).decode()
    return image_base64


def image2base64(image: np.ndarray, prefix="", use_rgb=False) -> str:
    """
    将图像编码为二进制字符串
    ``` python
        from io import BytesIO
        bgr_img = Image.fromarray(image)
        buff = BytesIO()
        mg.save(buff, format="PNG")
        image_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        image_base64 = str(base64.b64encode(image), encoding='utf-8')
    ```
    :param image: 图像
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :param use_rgb: True:输入image是RGB的图像, False:输入image是BGR格式的图像
    :return: 返回图像
    """
    # image = cv2.imencode('.jpeg', bgr_image)[1]
    # image_base64 = base64.b64encode(image).decode()
    # image_base64 = str(base64.b64encode(image), encoding='utf-8')
    img = image.copy()
    if len(img.shape) == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = prefix.split("/")
    ext = "." + ext[1] if len(ext) == 2 else ".png"
    img = cv2.imencode(ext, img)[1]
    bs64 = prefix + base64.b64encode(img).decode()
    return bs64


def base642image(image_bs64, use_rgb=False) -> np.ndarray:
    """
    將二进制字符串图像image_base64解码为正常形式
    :param image_bs64: 二进制字符串图像
    :param use_rgb: True:返回RGB的图像, False:返回BGR格式的图像
    :return: 返回图像矩阵
    """
    image_bs64 = bytes(image_bs64, 'utf-8')
    image = base64.b64decode(image_bs64)
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_image_base64(image_file, size=None):
    if not size:
        with open(image_file, 'rb') as f_in:
            image_base64 = base64.b64encode(f_in.read())
            image_base64 = str(image_base64, encoding='utf-8')
    else:
        bgr_image = read_image(image_file, size=size, use_rgb=False)
        image_base64 = image2base64(bgr_image)
    return image_base64


def bin2image(bin_data, size, norm=False, use_rgb=True):
    data = np.asarray(bytearray(bin_data), dtype="uint8")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    image = resize_image(image, size=size)
    if norm: image = image_normalize(image)
    return image


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def convert_anchor(anchors, height, width):
    """
    height, width, _ = image.shape
    :param win_name:
    :param anchors: <class 'tuple'>: (nums, 4)
    :return: boxes_list:[xmin, ymin, xmax, ymax]
    """
    boxes_list = []
    for index, anchor in enumerate(anchors):
        xmin = anchor[0] * width
        ymin = anchor[1] * height
        xmax = anchor[2] * width
        ymax = anchor[3] * height
        boxes_list.append([xmin, ymin, xmax, ymax])
    return boxes_list


def get_rect_crop_padding(image, rect, color=(0, 0, 0)):
    """
    :param image:
    :param rect:
    :param color: padding color value
    :return:
    """
    rect = [int(v) for v in rect]
    rows, cols = image.shape[:2]  # h,w,d
    x, y, width, height = rect
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(cols, x + width)  # 图像范围
    crop_y2 = min(rows, y + height)
    left_x = -x
    top_y = -y
    right_x = x + width - cols
    down_y = y + height - rows
    roi_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    # 只要存在边界越界的情况，就需要边界填充
    if top_y > 0 or down_y > 0 or left_x > 0 or right_x > 0:
        left_x = max(left_x, 0)
        right_x = max(right_x, 0)
        top_y = max(top_y, 0)
        down_y = max(0, down_y, 0)
        roi_image = cv2.copyMakeBorder(roi_image, int(top_y), int(down_y), int(left_x), int(right_x),
                                       cv2.BORDER_CONSTANT, value=color)
    return roi_image


def get_box_crop_padding(image, box, color=(0, 0, 0)):
    """
    :param image:
    :param box:
    :return:
    """
    rect = boxes2rects([box])[0]
    roi_image = get_rect_crop_padding(image, rect, color=color)
    return roi_image


def get_box_crop(image, box):
    """
    :param image:
    :param box:
    :return:
    """
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = box[:4]
    xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
    xmax, ymax = min(w, int(xmax)), min(h, int(ymax))
    roi = image[ymin:ymax, xmin:xmax]
    return roi


get_bbox_crop = get_box_crop


def get_bboxes_crop(image, bboxes):
    """
    已经废弃，使用get_boxes_crop代替
    :param image:
    :param bbox:
    :return:
    """
    crops = [get_box_crop(image, box) for box in bboxes]
    return crops


def get_boxes_crop(image, boxes):
    """
    :param image:
    :param boxes:
    :return:
    """
    crops = [get_box_crop(image, box) for box in boxes]
    return crops


def get_bboxes_crop_padding(image, bboxes, size=(), color=(0, 0, 0)):
    """
    已经废弃，使用get_boxes_crop_padding代替
    :param image:
    :param bboxes:
    :param size:
    :param color:
    :return:
    """
    rects = boxes2rects(bboxes)
    crops = []
    for rect in rects:
        roi = get_rect_crop_padding(image, rect, color=color)
        if size: roi = resize_image(roi, size=size)
        crops.append(roi)
    return crops


def get_boxes_crop_padding(image, boxes, size=(), color=(0, 0, 0)):
    """
    :param image:
    :param boxes:
    :param size:
    :param color:
    :return:
    """
    rects = boxes2rects(boxes)
    crops = []
    for rect in rects:
        roi = get_rect_crop_padding(image, rect, color=color)
        if size: roi = resize_image(roi, size=size)
        crops.append(roi)
    return crops


def get_rects_crop_padding(image, rects, size=(), color=(0, 0, 0)):
    """
    :param image:
    :param rects:
    :param size:
    :param color:
    :return:
    """
    crops = []
    for rect in rects:
        roi = get_rect_crop_padding(image, rect, color=color)
        if size: roi = resize_image(roi, size=size)
        crops.append(roi)
    return crops


def center_crop(image, crop_size=[112, 112]):
    """
    central_crop
    :param image: input numpy type image
    :param crop_size: (W,H) crop_size must less than x.shape[:2]=[crop_h,crop_w]
    :return:
    """
    h, w = image.shape[:2]
    y = int(round((h - crop_size[1]) / 2.))
    x = int(round((w - crop_size[0]) / 2.))
    y = max(y, 0)
    x = max(x, 0)
    return image[y:y + crop_size[1], x:x + crop_size[0]]


def center_crop_padding(image, crop_size, color=(0, 0, 0)):
    """
    :param image:
    :param crop_size: [crop_w,crop_h]
    :return:
    """
    h, w = image.shape[:2]
    y = int(round((h - crop_size[1]) / 2.))
    x = int(round((w - crop_size[0]) / 2.))
    rect = [x, y, crop_size[0], crop_size[1]]
    roi_image = get_rect_crop_padding(image, rect, color=color)
    return roi_image


def center_crop_padding_mask_shift(mask, size=(256, 256), center=True, scale=1.0, color=(0, 0, 0)):
    """
    实现将白色区域平移到中心，居中显示(区域相对大小不会变化)
    :param mask:
    :param size: [crop_w,crop_h]
    :param center: 是否居中
    :param scale: 是否缩放
    :param color: padding的颜色
    :return:
    """
    mask = center_crop_padding(mask, crop_size=size, color=color)
    if center:
        box = get_mask_boundrect_cv(mask, binarize=False, shift=0)
        if len(box) > 0:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            new = (cx - size[0] / 2, cy - size[1] / 2,
                   cx + size[0] / 2, cy + size[1] / 2)
            mask = get_box_crop_padding(mask, new, color=color)
    if scale < 1.0:
        mask = get_scale_image(mask, scale=scale, color=color)
    return mask


def center_crop_padding_mask_resize(mask, size=(256, 256), center=True, scale=1.0, color=(0, 0, 0)):
    """
    实现将白色区域调整到中心，居中显示(区域相对大小会变化)
    :param mask:
    :param size: [crop_w,crop_h]
    :param center: 是否居中
    :param scale: 是否缩放
    :param color: padding的颜色
    :return:
    """
    if center:
        box = get_mask_boundrect_cv(mask, binarize=False, shift=0)
        if box: mask = get_box_crop(mask, box)
    mask = resize_image_padding(mask, size=size, color=color, interpolation=cv2.INTER_LINEAR)
    if scale < 1.0:
        mask = get_scale_image(mask, scale=scale, color=color)
    return mask


def draw_yaws_pitchs_rolls_axis_in_image(image,
                                         yaws,
                                         pitchs,
                                         rolls,
                                         centers=None,
                                         size=75):
    """
    绘制yaw, pitch, roll坐标轴
    :param image:
    :param yaws:
    :param pitchs:
    :param rolls:
    :param centers:
    :param size:
    :return:
    """
    assert len(yaws) == len(pitchs)
    assert len(pitchs) == len(rolls)
    for i in range(len(yaws)):
        center = None if centers is None else centers[i]
        image = draw_yaw_pitch_roll_in_left_axis(image, yaws[i], pitchs[i], rolls[i], center, size=size)
    return image


def draw_yaw_pitch_roll_in_right_axis(image,
                                      yaw,
                                      pitch,
                                      roll,
                                      center=None,
                                      vis=True,
                                      size=75):
    """
    右手笛卡尔坐标：https://blog.csdn.net/a812073479/article/details/100103442
    pitch是围绕X轴旋转，也叫做俯仰角，左右耳朵连线构成X轴，如低头，点头动作
    yaw是围绕Y轴旋转，也叫偏航角，嘴巴鼻子连线构成Y轴，如左右摇头动作
    roll是围绕Z轴旋转，也叫翻滚角，眼睛和后脑勺连线构成Z轴，如歪头动作
    XYZ分别用红绿蓝表示，如(-10,0,0)表示绕X轴的反方向旋转10度
    :param image: BGR Image
    :param yaw:  绿色Y
    :param pitch: 红色X
    :param roll: 蓝色Z
    :param center:
    :param vis:
    :param size:
    :return:
    """
    text = "(pitch,yaw,roll)=({:3.3f},{:3.3f},{:3.3f})".format(pitch, yaw, roll)
    if center is None:
        h, w, c = image.shape
        center = (w / 2, h / 2)
    cx, cy = center
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    # X-Axis pointing to right. drawn in red
    x1 = cx - size * (cos(yaw) * cos(roll))
    y1 = cy - size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw))
    color_yaw_x = (0, 0, 255)  # BGR
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + cx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + cy
    color_pitch_y = (0, 255, 0)
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + cx
    y3 = size * (-cos(yaw) * sin(pitch)) + cy
    color_roll_z = (255, 0, 0)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x1), int(y1)), color_yaw_x, 2, tipLength=0.2)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x2), int(y2)), color_pitch_y, 2, tipLength=0.2)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x3), int(y3)), color_roll_z, 2, tipLength=0.2)
    if vis:
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, str(text),
                            # (int(cx), int(cy) + 10),
                            (10, 10),
                            fontFace,
                            fontScale=0.3,
                            color=(255, 0, 0),
                            thickness=1)
    return image


def draw_yaw_pitch_roll_in_left_axis(image,
                                     yaw,
                                     pitch,
                                     roll,
                                     center=None,
                                     vis=True,
                                     size=75):
    """
    左手坐标
    """
    text = "(pitch,yaw,roll)=({:3.3f},{:3.3f},{:3.3f})".format(pitch, yaw, roll)
    if center is None:
        h, w, c = image.shape
        center = (w / 2, h / 2)
    cx, cy = center
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + cx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + cy
    color_yaw_x = (0, 0, 255)  # BGR
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + cx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + cy
    color_pitch_y = (0, 255, 0)
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + cx
    y3 = size * (-cos(yaw) * sin(pitch)) + cy
    color_roll_z = (255, 0, 0)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x1), int(y1)), color_yaw_x, 2, tipLength=0.2)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x2), int(y2)), color_pitch_y, 2, tipLength=0.2)
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(x3), int(y3)), color_roll_z, 2, tipLength=0.2)
    if vis:
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, str(text),
                            # (int(cx), int(cy) + 10),
                            (10, 10),
                            fontFace,
                            fontScale=0.3,
                            color=(255, 0, 0),
                            thickness=1)
    return image


def polygons2boxes(polygons: List[np.ndarray]):
    """
    将多边形轮廓转转为矩形框
    :param self:
    :param polygons: [num_polygons,num_point,2]
    :return: boxes:[num_polygons,4], box is [xmin, ymin, xmax, ymax]
    """
    boxes = []
    for pts in polygons:
        if not isinstance(pts, np.ndarray): pts = np.asarray(pts)
        xmin = min(pts[:, 0])
        ymin = min(pts[:, 1])
        xmax = max(pts[:, 0])
        ymax = max(pts[:, 1])
        boxes.append([xmin, ymin, xmax, ymax])
    boxes = np.asarray(boxes)
    return boxes


points2bbox = polygons2boxes
points2boxes = polygons2boxes


def boxes2polygons(boxes: np.ndarray or List[np.ndarray]):
    """
    将矩形框转为多边形轮廓
    :param self:
    :param boxes:[num_polygons,4], box is [xmin, ymin, xmax, ymax]
    :return: polygons: [num_polygons,num_point,2]
    """
    polygons = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        polygons.append(pts)
    polygons = np.asarray(polygons)
    return polygons


def get_mask_iou(mask1, mask2, binarize=True):
    """
    计算两个Mask的IOU
    h, w = mask1.shape[:2]
    mask2 = cv2.resize(mask2, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    :param mask1:
    :param mask2:
    :param binarize:
    :return:
    """
    if binarize:
        ret, mask1 = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, mask2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inter = cv2.bitwise_and(mask1, mask2)  # 交集
    union = cv2.bitwise_or(mask1, mask2)  # 并集
    # 测试发现np.float32会比np.int32快
    # inter_area = np.sum(inter > 0)  # inter>0
    # union_area = np.sum(union > 0)  # union>0
    inter_area = np.sum(np.float32(np.greater(inter, 0)))
    union_area = np.sum(np.float32(np.greater(union, 0)))
    iou = inter_area / max(union_area, 1e-8)
    return iou


def get_mask_iou1(mask1, mask2, binarize=True):
    """
    比get_mask_iou慢一点
    :param mask1:
    :param mask2:
    :param binarize:
    :return:
    """
    if binarize:
        ret, mask1 = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, mask2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask1 = mask1 >= 255
    mask2 = mask2 >= 255
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = mask1 * mask2
    inter_ = (inter == 1).sum()  # 交集
    iou = inter_ / max((area1 + area2 - inter_), 1e-8)
    return iou


def get_contours_iou(contour1, contour2, image_size: Tuple = None, plot=False):
    """
    计算两个轮廓(多边形)交并比(Intersection-over-Union,IoU)
    :param contour1: 多边形1 (num_points,2),由num_points个点构成的封闭多边形
    :param contour2: 多边形2 (num_points,2),由num_points个点构成的封闭多边形
    :param image_size: (W,H) image size,用于可视化,不会影响contours,iou的结果
    :param plot: 是否可视化Mask
    :return: contours: 多边形1和多边形2重叠区域
             iou: 多边形1和多边形2的交并比
    """
    contour1 = np.asarray(contour1, dtype=np.int32)
    contour2 = np.asarray(contour2, dtype=np.int32)
    if image_size:
        w, h = image_size
        xmin, ymin = (0, 0)
    else:
        xmin = min(min(contour1[:, 0]), min(contour2[:, 0]))
        ymin = min(min(contour1[:, 1]), min(contour2[:, 1]))
        contour1 = contour1 - (xmin, ymin)
        contour2 = contour2 - (xmin, ymin)
        w = max(max(contour1[:, 0]), max(contour2[:, 0])) + 1
        h = max(max(contour1[:, 1]), max(contour2[:, 1])) + 1
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    mask = np.zeros(shape=(h, w), dtype=np.uint8)
    mask1 = np.zeros(shape=(h, w), dtype=np.uint8)
    mask2 = np.zeros(shape=(h, w), dtype=np.uint8)
    mask1 = cv2.fillPoly(mask1, [contour1], color=55)
    mask2 = cv2.fillPoly(mask2, [contour2], color=200)
    mask[(mask1 + mask2) == 255] = 255
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    area = sum([cv2.contourArea(c) for c in contours])
    contours = [c.reshape(-1, 2) + (xmin, ymin) for c in contours]
    # area = np.sum(mask > 0)
    iou = area / max((area1 + area2 - area), 1e-8)
    if plot:
        print("U(mask1,mask2)={}".format(iou))
        cv_show_image("mask1", mask1, delay=1)
        cv_show_image("mask2", mask2, delay=1)
        cv_show_image("mask1+mask2", mask1 + mask2, delay=1)
        cv_show_image("U(mask1,mask2)={:3.3f}".format(iou), mask, delay=0)
    return contours, iou


def get_image_mask(image: np.ndarray, inv=False):
    """获得图像的mask"""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if inv:
        ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def get_mask_erode_dilate(mask, ksize, binarize=False):
    """
    获得膨胀和腐蚀的mask
    :param mask:
    :param ksize:
    :param binarize:
    :return:
    """
    if binarize: mask = get_image_mask(mask, inv=False)
    if ksize > 0:
        k = np.ones((5, 5), np.uint8)  # 设置kenenel大小
        mask = cv2.erode(mask, k)  # 腐蚀去除白点
    if ksize < 0:
        k = np.ones((5, 5), np.uint8)  # 设置kenenel大小
        mask = cv2.dilate(mask, k)  # 膨胀增大白点
    return mask


def get_mask_morphology(image, ksize, binarize=False, op=cv2.MORPH_OPEN, itera=1):
    """
    形态学计算
    :param image: 可以是二值化图像，也可以是RGB图像
    :param ksize:
    :param binarize:
    :param op:为形态变换的类型，包括如下取值类型：
            MORPH_ERODE：腐蚀，与调用腐蚀函数erode效果相同
            MORPH_DILATE：膨胀，与调用膨胀函数dilate效果相同
            MORPH_OPEN：开运算，对图像先进行腐蚀再膨胀，等同于dilate(erode(src,kernal))，开运算对图像的边界进行平滑、去掉凸起等
            MORPH_CLOSE：闭运算，对图像先进行膨胀在腐蚀，等同于erode(dilate(src,kernal))，闭运算用于填充图像内部的小空洞、填充图像的凹陷等
            MORPH_GRADIENT：梯度图，用膨胀图减腐蚀图，等同于dilate(src,kernal)−erode(src,kernal)，可以用于获得图像中物体的轮廓
            MORPH_TOPHAT：顶帽，又称礼帽，用原图像减去开运算后的图像，等同于src−open(src,kernal)，可以用于获得原图像中比周围亮的区域
            MORPH_BLACKHAT：黑帽，闭运算图像减去原图像，等同于close(src,kernal)−src，可以用于获取原图像中比周围暗的区域
            MORPH_HITMISS：击中击不中变换，用于匹配处理图像中是否存在核对应的图像，匹配时，需要确保核矩阵非0部分和为0部分都能匹配，注意该变换只能处理灰度图像。
    :param itera:
    :return:
    """
    if binarize: image = get_image_mask(image, inv=False)
    kernel = np.ones((ksize, ksize), np.uint8)
    # morphologyEx输入的image可以是RGB图像
    image = cv2.morphologyEx(image, op=op, kernel=kernel, iterations=itera)
    return image


def get_scale_image(image, scale=0.85, offset=(0, 0), color=(0, 0, 0), interpolation=cv2.INTER_NEAREST):
    """
    同比居中缩小image，以便居中显示
    :param image: mask
    :param scale: 缩放比例
    :param offset: 偏移量(x,y)
    :return: 返回缩放的轮廓Mask
    """
    h, w = image.shape[:2]
    bg = create_image(image.shape, color=color)
    fg = cv2.resize(image, dsize=(int(w * scale), int(h * scale)), interpolation=interpolation)
    sh, sw = fg.shape[:2]
    xmin = (w - sw) // 2 + offset[0]
    ymin = (h - sh) // 2 + offset[1]
    bg[ymin:ymin + sh, xmin:xmin + sw] = fg
    return bg


get_scale_mask = get_scale_image


def get_scale_contours(contours, size, scale=0.85, offset=(0, 0)):
    """
    同比居中缩小mask的轮廓，以便居中显示
    :param contours:  mask的轮廓
    :param size: 输出大小(W,H)
    :param scale: 缩放比例
    :param offset: 偏移量(x,y)
    :return: 返回缩放的轮廓contours
    """
    dst_contours = copy.deepcopy(contours)
    sw, sh = (int(size[0] * scale), int(size[1] * scale))
    xmin = (size[0] - sw) // 2
    ymin = (size[1] - sh) // 2
    for i in range(len(dst_contours)):
        for c in range(len(dst_contours[i])):
            d = dst_contours[i][c] * scale + (xmin, ymin) + offset
            dst_contours[i][c] = np.asarray(d, dtype=np.int32)
    return dst_contours


def resize_image_points(image: np.ndarray, points, size):
    """
    缩放图片，并缩放对应的points
    :param image: 图片
    :param points: shape is (N,2)
    :param size:
    :return:
    """
    height, width = image.shape[:2]
    image = resize_image(image, size=size)
    resize_height, resize_width = image.shape[:2]
    scale = [resize_width / width, resize_height / height]
    if len(points) > 0:
        if points.shape[1] > 2:  # 如果points是多维度
            scale += [0] * (points.shape[1] - 2)
        points = np.asarray(points * scale)
    return image, points


def pointPolygonTest(point, contour, measureDist=False):
    """
    此功能可查找图像中的点与轮廓之间的最短距离.
    当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零
    :param point: (x,y)
    :param contour: np.asarray,(num_point,2)
    :param measureDist: 如果为True，则查找签名距离(signed distance)
                        如果为False，则查找该点是在内部还是外部或在轮廓上（它分别返回+1，-1,0）
    如您不想找到距离，请设置measureDist=False，因为这是一个耗时的过程. 因此，将其设为False可提供2-3倍的加速.
    :return:
    """
    contour = np.asarray(contour, dtype=np.int32)
    point = (int(point[0]), int(point[1]))
    dist = cv2.pointPolygonTest(contour, point, measureDist)
    return dist


def draw_image_contours(image, contours: List[np.ndarray], texts=[], color=(), alpha=0.5, thickness=1, fontScale=0.8,
                        drawType="ch"):
    """
    参考：draw_image_mask_color
    :param image:
    :param contours: List[np.ndarray],每个列表是一个轮廓(num_points,1,2)
    :param texts:轮廓文本
    :param color:绘制轮廓的颜色
    :param alpha:绘制颜色的透明度
    :param thickness:轮廓线宽
    :return:
    """
    for i in range(0, len(contours)):
        c = color if color else color_table[i + 1]
        t = str(texts[i]) if texts else ""
        p = np.asarray(contours[i], dtype=np.int32)
        b = (min(p[:, 0]), min(p[:, 1]), max(p[:, 0]), max(p[:, 1]))
        if len(p.shape) == 2: p = [p]
        image[:] = cv2.drawContours(image, p, contourIdx=-1, color=c, thickness=thickness)
        bgimg = image.copy()
        bgimg = cv2.fillPoly(bgimg, p, color=c)
        image = cv2.addWeighted(src1=image, alpha=1 - alpha, src2=bgimg, beta=alpha, gamma=0)
        if t: image = draw_text(image, point=(b[0], b[1]), color=c, text=t, thickness=thickness,
                                fontScale=fontScale, drawType=drawType)
    return image


draw_image_mask_color = color_utils.draw_image_mask_color


def draw_mask_contours(contours: List[np.ndarray], size, value=255):
    """
    参考：draw_image_mask_color
    :param contours: List[np.ndarray],每个列表是一个轮廓(num_points,1,2)
    :return:
    """
    mask = np.zeros(shape=(size[1], size[0]), dtype=np.uint8)
    for i in range(0, len(contours)):
        p = np.asarray(contours[i], dtype=np.int32)
        if len(p.shape) == 2: p = [p]
        mask[:] = cv2.drawContours(mask, p, contourIdx=-1, color=value, thickness=-1)
    return mask


def contours_interpolation(contours, n=1000):
    """
    对图像轮廓或者对直线进行线性差值，来源于ultralytics/utils/ops.py
    轮廓插值，线性插值，直线插值
    Inputs a list of contours (n,2) and returns a list of contours (n,2) up-sampled to n points each.

    Args:
        contours (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        contours (list): the resampled contours.
    """
    for i, s in enumerate(contours):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        contours[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return contours


def get_mask_boundrect_cv(mask, binarize=False, shift=0):
    """
    获得mask的最大外接矩形框(其速度比get_mask_boundrect快2倍左右)
    :param mask:
    :param binarize: 是否对mask进行二值化
    :param shift: 矩形框偏移量,shift>0表示扩大shift个像素，shift<0表示缩小shift个像素，默认为0不偏移
    :return: box=[xmin,ymin,xmax,ymax]
    """
    h, w = mask.shape[:2]
    if binarize:
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours = find_mask_contours(mask)
    if len(contours) == 0: return []
    contours = np.concatenate(contours)
    xmin = np.min(contours[:, 0])
    ymin = np.min(contours[:, 1])
    xmax = np.max(contours[:, 0])
    ymax = np.max(contours[:, 1])
    xmin = max(0, int(xmin - shift))
    ymin = max(0, int(ymin - shift))
    xmax = min(w, int(xmax + shift))
    ymax = min(h, int(ymax + shift))
    return [xmin, ymin, xmax, ymax]


def binarize(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
    """
    二值化
    :param mask:
    :param thresh:
    :param maxval:
    :param type:
    :return:
    """
    ret, mask = cv2.threshold(mask, thresh=thresh, maxval=maxval, type=type)
    return mask


def get_mask_boundrect(mask, binarize=False, shift=0):
    """
    获得mask的最大外接矩形框(比较慢)
    :param mask:
    :param binarize: 是否对mask进行二值化
    :param shift: 矩形框偏移量,shift>0表示扩大shift个像素，shift<0表示缩小shift个像素，默认为0不偏移
    :return: box=[xmin,ymin,xmax,ymax]
    """

    def get_argminmax(v):
        """获得首个和最后一个最大值的index"""
        v = np.asarray(v)
        max1 = np.argmax(v)
        max2 = len(v) - np.argmax(v[::-1]) - 1
        return max1, max2

    h, w = mask.shape[:2]
    if binarize:
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    y = np.sum(mask, axis=1)  # 将图像往Y轴累加投影
    x = np.sum(mask, axis=0)  # 将图像往X轴累加投影
    if np.sum(y) == 0 or np.sum(x) == 0: return []
    y = y > 0
    x = x > 0
    ymin, ymax = get_argminmax(y)
    xmin, xmax = get_argminmax(x)
    xmin = max(0, int(xmin - shift))
    ymin = max(0, int(ymin - shift))
    xmax = min(w, int(xmax + shift))
    ymax = min(h, int(ymax + shift))
    return [xmin, ymin, xmax, ymax]


def get_mask_boundrect_low(mask, binarize=False, shift=0):
    """
    获得mask的最大外接矩形框(比较慢)
    :param mask:
    :param binarize: 是否对mask进行二值化
    :param shift: 矩形框偏移量,shift>0表示扩大shift个像素，shift<0表示缩小shift个像素，默认为0不偏移
    :return: box=[xmin,ymin,xmax,ymax]
    """

    h, w = mask.shape[:2]
    if binarize:
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    y_idxes, x_idxes = mask.nonzero()
    if len(y_idxes) == 0 or len(x_idxes) == 0: return []
    xmin = x_idxes.min()
    xmax = x_idxes.max()
    ymin = y_idxes.min()
    ymax = y_idxes.max()
    xmin = max(0, int(xmin - shift))
    ymin = max(0, int(ymin - shift))
    xmax = min(w, int(xmax + shift))
    ymax = min(h, int(ymax + shift))
    return [xmin, ymin, xmax, ymax]


def find_mask_contours(mask, max_nums=-1, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE):
    """
    寻找一个二值Mask图像的轮廓
    cv2.findContours(mask，mode, method)
    mask: 输入mask必须是二值Mask图像，注意黑色表示背景，白色表示物体，即在黑色背景里寻找白色物体的轮廓
    mode:
        cv.RETR_LIST: 提取所有轮廓
        cv.RETR_EXTERNAL: 只提取外轮廓（可能有多个外轮廓，只是不提取孔轮廓）
        cv.RETR_TREE: 树形结构表示轮廓的从属关系
        cv.RETR_CCOMP: 提取所有轮廓，把它们组织成两级结构。
                    第一级是连通域的外边界(external boundaries)，
                    第二级是孔边界(boundaries of holes)。如果在孔中间还有另外的连通域，则被当成另一个外边界
    method:
        CHAIN_APPROX_NONE  :存储所有的轮廓点
        CHAIN_APPROX_SIMPLE:压缩水平，垂直和对角线段，只留下端点,例如矩形轮廓可以用4个点编码。
    :param mask: 输入mask必须是二值Mask图像，注意黑色表示背景，白色表示物体，即在黑色背景里寻找白色物体的轮廓
    :param max_nums: 按照面积大小进行排序，返回max_nums个轮廓
    :return: contours List[np.ndarray(num_point,2)] 返回二值Mask图像的轮廓
    """
    contours, hierarchy = cv2.findContours(mask, mode=mode, method=method)
    contours = [c.reshape(-1, 2) for c in contours]
    # contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    contours = sorted(contours, key=lambda c: len(c), reverse=True)
    if max_nums > 0:
        max_nums = min(max_nums, len(contours))
        contours = contours[0:max_nums]
    return contours


def find_minAreaRect(contours, order=True):
    """
    获得旋转矩形框，即最小外接矩形的四个角点
    :param contours: [shape(n,2),...,]
    :param order: 对4个点按顺时针方向进行排序:[top-left, top-right, bottom-right, bottom-left]
    :return:
    """
    points = []
    for i in range(len(contours)):
        pts = corner_utils.get_obb_points(pts=contours[i], order=order)
        points.append(pts)
    return points


def find_image_contours(mask: np.ndarray, target_label: List[int] = [1, 2]) -> List[List[np.ndarray]]:
    """
    寻找一个多值Mask图像的轮廓
    :param mask: np.ndarray：多值Mask图像
    :param target_label:  List[int]，查找区域数值等于target_label的目标轮廓，可指定多个，
                         为空或这None时，表示查找大于0的轮廓
    :return: 返回多值Mask图像的轮廓
    """
    if not target_label:
        return [find_mask_contours(mask)]
    # cv2.findContours(mask，mode, method)寻找一个二值Mask的轮廓， 输入mask必须是二值图像，不能直接使用
    contours = []
    for label in target_label:
        m = np.zeros_like(mask, dtype=np.uint8)
        m[mask == label] = 255  # 转换为二值Mask
        contour = find_mask_contours(m)
        contours.append(contour)
    return contours


def get_image_points_valid_range(image, points, valid_range, crop=True, color=(255, 255, 255)):
    """
    获得在valid_range范围的points
    :param image: 图像
    :param points: 点集合shape: (N,2)
    :param valid_range: 有效范围(xmin,ymin,xmax,ymax)
    :param crop: 是否采裁剪图像在valid_range有效区域
    :param color: crop图像时，填充颜色
    :return:
    """
    valid_point, valid_index = get_points_valid_range(points, valid_range)
    if crop:
        image = get_box_crop_padding(image, valid_range, color=color)
        valid_point = valid_point - (valid_range[0], valid_range[1])
    return image, valid_point, valid_index


def get_points_valid_range(points, valid_range):
    """
    获得在valid_range范围的points
    :param points: 点集合shape: (N,2)
    :param valid_range: 有效范围(xmin,ymin,xmax,ymax)
    :return:
    """
    xmin, ymin, xmax, ymax = valid_range
    contour = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
    valid_point = []
    valid_index = []
    for i in range(len(points)):
        pt = points[i, :]
        dist = pointPolygonTest(pt, contour)
        if dist > 0:
            valid_point.append(pt)
            valid_index.append(i)
    valid_point = np.asarray(valid_point)
    valid_index = np.asarray(valid_index)
    return valid_point, valid_index


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def addMouseCallback(winname, param, callbackFunc=None, info="%"):
    """
     添加点击事件
    :param winname:
    :param param:
    :param callbackFunc:
    :return:
    """
    cv2.namedWindow(winname)

    def default_callbackFunc(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("(x,y)=({},{}),".format(x, y) + info % param[y][x])

    if callbackFunc is None:
        callbackFunc = default_callbackFunc
    cv2.setMouseCallback(winname, callbackFunc, param)


class EventCv():
    def __init__(self):
        self.image = None

    def update_image(self, image):
        self.image = image

    def callback_print_image(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("(x,y)=({},{}),data={}".format(x, y, self.image[y][x]))

    def add_mouse_event(self, winname, param=None, callbackFunc=None):
        """
         添加点击事件
        :param winname:
        :param param:
        :param callbackFunc:
        :return:
        """
        cv2.namedWindow(winname)
        if callbackFunc is None:
            callbackFunc = self.callback_print_image
        cv2.setMouseCallback(winname, callbackFunc, param=param)


def get_image_block(image, grid=[3, 3], same=False):
    """
    :param image:
    :param grid: cell grid
    :param same: True : 保证每个cell的大小相同,会裁剪掉部分像素
           same: False: 及可能每个cell的大小相同,边界的cell会偏大
    :return:
    """
    image_block = []
    grid_w, grid_h = grid
    height, width, channel = np.shape(image)  # H,W,C
    step_w = int(height / grid_w)
    step_h = int(width / grid_h)
    for i in range(0, grid_h):
        for j in range(0, grid_w):
            x1 = j * step_w
            y1 = i * step_h
            x2 = (j + 1) * step_w
            y2 = (i + 1) * step_h
            if same:
                block = image[y1:y2, x1:x2]
            else:
                x2 = x2 if x2 + step_w < width else width
                y2 = y2 if y2 + step_h < height else height
                block = image[y1:y2, x1:x2]
            cv_show_image("block", block)
            image_block.append(block)
    return image_block


def image_composite(image: np.ndarray, alpha: np.ndarray, bg_img=(219, 142, 67)):
    """
    图像融合：合成图 = 前景*alpha+背景*(1-alpha)
    https://blog.csdn.net/guduruyu/article/details/71439733
    更有效的C++实现: https://www.aiuai.cn/aifarm1237.html
    :param image: RGB图像(uint8)
    :param alpha: 单通道的alpha图像(uint8)
    :param bg_img: 背景图像,可以是任意的分辨率图像，也可以指定指定纯色的背景
    :return: 返回与背景合成的图像
    """
    if isinstance(bg_img, tuple) or isinstance(bg_img, list):
        bg = np.zeros_like(image, dtype=np.uint8)
        bg_img = np.asarray(bg[:, :, 0:3] + bg_img, dtype=np.uint8)
    if len(alpha.shape) == 2:
        # alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        alpha = alpha[:, :, np.newaxis]
    if alpha.dtype == np.uint8:
        alpha = np.asarray(alpha / 255.0, dtype=np.float32)
    sh, sw, d = image.shape
    bh, bw, d = bg_img.shape
    ratio = [sw / bw, sh / bh]
    ratio = max(ratio)
    if ratio > 1:
        bg_img = cv2.resize(bg_img, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)))
    bg_img = bg_img[0: sh, 0: sw]
    image = image * alpha + bg_img * (1 - alpha)
    image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    return image


def frames2gif_by_imageio(frames, gif_file="test.gif", fps=2, loop=0, use_rgb=False):
    """
    pip install imageio 文件大，但质量较好
    :param frames:
    :param gif_file: 输出的GIF图的路径
    :param fps: 刷新频率
    :param loop: 循环次数
    :param use_rgb: frames是RGB格式，需要是BGR格式，use_bgr=True
    :return:
    """
    import imageio
    "(in ms) instead, e.g. `fps=50` == `duration=20` (1000 * 1/50)."
    duration = 1000 * 1 / fps
    # writer = imageio.get_writer(uri=gif_file, mode='I', fps=fps, loop=loop)
    writer = imageio.get_writer(uri=gif_file, mode='I', duration=duration, loop=loop)
    for image in frames:
        if use_rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer.append_data(image)
    writer.close()
    # imageio.mimwrite(out_gif_path, frames, fps=20)


def frames2gif_by_pil(frames, gif_file="test.gif", fps=2, loop=0, use_rgb=False):
    """
    https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    PS :loop=0表示无限循环，loop=n表示循环n+1次，如要循环一次，需要去掉loop参数(很奇葩吧！！！)，文件小，但质量较差
    :param frames:
    :param gif_file: 输出的GIF图的路径
    :param fps: 刷新频率
    :param loop: 循环次数
    :return:
    """
    if use_rgb: frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]
    images = [Image.fromarray(img) for img in frames]
    if loop == 1:
        images[0].save(gif_file, save_all=True, append_images=images[1:], duration=1000 / fps,
                       optimize=False)
    else:
        loop = max(0, loop - 1)
        images[0].save(gif_file, save_all=True, append_images=images[1:], duration=1000 / fps,
                       optimize=False, loop=loop)


def image_dir2gif(image_dir, size=None, gif_file=None, interval=1, fps=4, loop=0, padding=False, crop=[], use_pil=True):
    if not gif_file: gif_file = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + ".gif")
    file_list = file_utils.get_images_list(image_dir)
    image_file2gif(file_list, size=size, gif_file=gif_file, interval=interval, fps=fps, loop=loop,
                   padding=padding, crop=crop, use_pil=use_pil)


def image_file2gif(file_list, size=None, gif_file="test.gif", interval=1, fps=4, loop=0,
                   padding=True, crop=[], use_pil=False):
    """
    pip install imageio
    uri：合成后的gif动图的名字，可以随意更改。
    mode：操作模式，I表示多图，不用更改。
    fps：帧率，也就是画面每秒传输帧数，值越大，gif动图的播放速度越大。
    Usge:
        image_dir="path/to/image-directory"
        image_list = file_processing.get_files_lists(image_dir)
        image_file_list2gif(image_list, out_gif_path=out_gif_path, fps=fps)
    :param file_list:图片列表
    :param size:gif图片的大小
    :param gif_file: 输出的GIF图的路径
    :param fps: 刷新频率
    :param crop: 裁剪(xmin,ymin,xmax,ymax)
    :param loop: 循环次数
    :param use_pil: True使用PIL库生成GIF图，文件小，但质量较差
                    False使用imageio库生成GIF图，文件大，但质量较好
    :return:
    """
    frames = []
    for count, file in enumerate(file_list):
        if count % interval == 0:
            bgr = cv2.imread(file)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if len(crop) == 4: image = get_box_crop(image, box=crop)
            if padding:
                image = resize_image_padding(image, size=size)
            else:
                image = resize_image(image, size=size)
            frames.append(image)
    if use_pil:
        frames2gif_by_pil(frames, gif_file, fps=fps, loop=loop)
    else:
        frames2gif_by_imageio(frames, gif_file, fps=fps, loop=loop)


def get_video_capture(video, width=None, height=None, fps=None):
    """
     获得视频读取对象
     --   7W   Pix--> width=320,height=240
     --   30W  Pix--> width=640,height=480
     720P,100W Pix--> width=1280,height=720
     960P,130W Pix--> width=1280,height=1024
    1080P,200W Pix--> width=1920,height=1080
    :param video: video file or Camera ID
    :param width:   图像分辨率width
    :param height:  图像分辨率height
    :param fps:  设置视频播放帧率
    :return:
    """
    video_cap = cv2.VideoCapture(video)
    # 设置分辨率
    if width:
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        video_cap.set(cv2.CAP_PROP_FPS, fps)
    return video_cap


def get_video_info(video_cap: cv2.VideoCapture, vis=True):
    """
    获得视频的基础信息
    :param video_cap:视频对象
    :return:
    """
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    if vis: print("read video:width:{},height:{},fps:{},num_frames:{}".format(width, height, fps, num_frames))
    return width, height, num_frames, fps


def get_video_writer(video_file, width, height, fps):
    """
    获得视频存储对象
    :param video_file: 输出保存视频的文件路径
    :param width:   图像分辨率width
    :param height:  图像分辨率height
    :param fps:  设置视频播放帧率
    :return:
    """
    if not os.path.exists(os.path.dirname(video_file)):
        os.makedirs(os.path.dirname(video_file))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameSize = (int(width), int(height))
    video_writer = cv2.VideoWriter(video_file, fourcc, int(fps), frameSize)
    print("save video:width:{},height:{},fps:{}".format(width, height, fps))
    return video_writer


if __name__ == "__main__":
    # from utils import image_utils

    input_size = [int(490 / 2), int(800 / 2)]
    image_path = "test.jpg"
    src_boxes = [[8.20251, 1, 242.2412, 699.2236],
                 [201.14865, 204.18265, 468.605, 696.36163]]
    src_boxes = np.asarray(src_boxes)
    image = read_image(image_path)  # (800, 490, 3)
    image1, boxes1 = image_boxes_resize_padding(image, input_size, src_boxes)
    image1 = show_image_boxes("Det", image1, boxes1, color=(255, 0, 0), delay=3)
    boxes = image_boxes_resize_padding_inverse((image.shape[1], image.shape[0]), input_size, boxes1)
    show_image_boxes("image", image, boxes)
