# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-03-22 09:11:35
    @Brief  :
"""
import sys
import os
import cv2
import base64
import numpy as np
from typing import Any


def base642image(image_base64, prefix="image", use_rgb=False) -> np.ndarray:
    """
    將二进制字符串解码为图像
    :param image_base64: 二进制字符串图像
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :param use_rgb: True:返回RGB的图像, False:返回BGR格式的图像
    :return: 返回图像
    """
    image_base64 = image_base64[len(prefix):]
    image_base64 = bytes(image_base64, 'utf-8')
    image = base64.b64decode(image_base64)
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image2base64(image: np.ndarray, prefix="image", use_rgb=False) -> str:
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
    :param use_rgb: True:输入image是RGB的图像, False:返输入image是BGR格式的图像
    :return: 返回图像
    """
    img = image.copy()
    if len(img.shape) == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imencode('.jpeg', img)[1]
    image_base64 = prefix + base64.b64encode(img).decode()
    return image_base64


def read_file2base64(file, prefix="image") -> str:
    """
    读取文件,并编码为二进制字符串
    :param file: 文件路径
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :return:返回二进制字符串
    """
    file_base64 = prefix + base64.b64encode(open(file, 'rb').read()).decode()
    return file_base64


def array2base64(data: Any, prefix="image", use_rgb=False) -> Any:
    if isinstance(data, np.ndarray) and data.dtype == np.uint8:
        return image2base64(data, prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = array2base64(data[i], prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = array2base64(v, prefix=prefix, use_rgb=use_rgb)
    return data


def base642array(data: Any, prefix="image", use_rgb=False) -> Any:
    if isinstance(data, str) and prefix == data[0:len(prefix)]:
        return base642image(data, prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = base642array(data[i], prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = base642array(v, prefix=prefix, use_rgb=use_rgb)
    return data


def cv_show_image(title: str, image: np.ndarray, use_rgb=True, delay=0):
    """
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入是否是RGB图像
    :param use_rgb: True:输入image是RGB的图像, False:返输入image是BGR格式的图像
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


if __name__ == "__main__":
    file = "/home/dm/project/python-learning-notes/utils/test.jpg"
    bgr1 = cv2.imread(file)
    image_base64 = image2base64(bgr1)
    image_base64 = read_file2base64(file)
    bgr2 = base642image(image_base64, use_rgb=False)
    cv_show_image("bgr1", bgr1, use_rgb=False, delay=1)
    cv_show_image("bgr2", bgr2, use_rgb=False)
