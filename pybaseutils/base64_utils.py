# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import sys
import os

import PIL.Image as Image
import cv2
import base64
import numpy as np
from typing import Any

IMG_PREFIX = "image/jpg"  # 图片base64字符串前缀
precision = 6  # 小数点精度


def isbase64(data: str, prefix=IMG_PREFIX):
    """判断是否是二进制字符串图像"""
    return prefix == data[0:len(prefix)]


def base642image(bs64, prefix=IMG_PREFIX, use_rgb=False) -> np.ndarray:
    """
    將二进制字符串解码为图像
    :param bs64: 二进制字符串图像
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :param use_rgb: True:返回RGB的图像, False:返回BGR格式的图像
    :return: 返回图像
    """
    if prefix == bs64[0:len(prefix)]:
        bs64 = bs64[len(prefix):]
    bs64 = bytes(bs64, 'utf-8')
    image = base64.b64decode(bs64)
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, flags=cv2.IMREAD_UNCHANGED)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image2base64(image: np.ndarray, prefix=IMG_PREFIX, use_rgb=False) -> str:
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
    img = image.copy()
    if len(img.shape) == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = prefix.split("/")
    # ext = "." + ext[1] if len(ext) == 2 else ".png" # TODO libpng error: bad parameters to zlib
    # ext = "." + ext[1] if len(ext) == 2 else ".jpg"
    img = cv2.imencode('.jpg', img)[1]
    bs64 = prefix + base64.b64encode(img).decode()
    return bs64


def file2base64(file, prefix=IMG_PREFIX) -> str:
    """
    将文件编码为base64字符串
    :param file: 文件路径
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :return:base64字符串
    """
    bs64 = prefix + base64.b64encode(open(file, 'rb').read()).decode("utf-8")
    return bs64


def base642file(file, bs64, prefix=IMG_PREFIX) -> str:
    """
    将base64字符串解码为文件
    :param file: 文件路径
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :return:文件路径
    """
    if prefix == bs64[0:len(prefix)]:
        bs64 = bs64[len(prefix):]
    bs64 = base64.b64decode(bs64)
    with open(file, 'wb') as f: f.write(bs64)
    return file


def array2base64(data: Any, prefix=IMG_PREFIX, use_rgb=False) -> Any:
    """
    序列化:将输入数据含有图像数据(ndarray)都编码为base64字符串
    :param data: 输入数据
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :param use_rgb: True:输入image是RGB的图像, False:输入image是BGR格式的图像
    :return:
    """
    if isinstance(data, np.ndarray) and data.dtype == np.uint8:
        return image2base64(data, prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, Image.Image):
        return image2base64(np.asarray(data), prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return round(float(data), precision)
    elif isinstance(data, float):
        return round(data, precision)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = array2base64(data[i], prefix=prefix, use_rgb=use_rgb)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = array2base64(v, prefix=prefix, use_rgb=use_rgb)
    return data


def base642array(data: Any, prefix=IMG_PREFIX, use_rgb=False, img_type="numpy") -> Any:
    """
    反序列化:将输入数据含有base64字符串都解码为图像数据(ndarray)
    :param data: 输入数据
    :param prefix: base64字符串前缀,用于表识字符串的类型
    :param use_rgb: True:输入image是RGB的图像, False:输入image是BGR格式的图像
    :param img_type: numpy:返回numpy图像, pil:返回PIL图像
    :return:
    """
    if isinstance(data, str) and prefix == data[0:len(prefix)]:
        data = base642image(data, prefix=prefix, use_rgb=use_rgb)  # numpy image
        if img_type == "pil": data = Image.fromarray(data)
        return data
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = base642array(data[i], prefix=prefix, use_rgb=use_rgb, img_type=img_type)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = base642array(v, prefix=prefix, use_rgb=use_rgb, img_type=img_type)
    return data


serialization = array2base64  # 序列化
deserialization = base642array  # 反序列化
if __name__ == "__main__":
    from pybaseutils import file_utils, image_utils

    image_dir = "/home/PKing/Downloads/image"
    image_list = file_utils.get_images_list(image_dir)
    for image_file in image_list:
        print(image_file)
        src = image_utils.read_image(image_file, use_rgb=True)
        image_base64 = image2base64(src, prefix="data:image/jpeg;base64,", )
        img = Image.fromarray(src)
        data = {"image": img, "file": image_file}
        data = serialization(data)
        data = deserialization(data)
        dst = data["image"]
        image_utils.show_image("src", src, delay=0)
