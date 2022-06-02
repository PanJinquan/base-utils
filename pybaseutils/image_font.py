# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pybaseutils import image_utils, file_utils
from pybaseutils.font_style import FONT_TYPE

FONT_ROOT = os.path.join(os.path.dirname(__file__), "font_style")


def draw_image_text(bgr, point, text, style="楷体", size=20, color=(255, 255, 255)):
    """
    在图像中显示汉字
    https://blog.csdn.net/weixin_44237337/article/details/119817801
    :param bgr: 字体背景
    :param point:
    :param text:
    :param style: 字体风格
    :param size:
    :param color:
    :return:
    """
    bgr = Image.fromarray(bgr)
    draw = ImageDraw.Draw(bgr)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    # font = ImageFont.truetype("simhei.ttf", size=size, encoding="utf-8")
    font_type = ImageFont.truetype(os.path.join(FONT_ROOT, FONT_TYPE[style]), size=size, encoding="utf-8")
    draw.text(point, text, fill=color, font=font_type)
    bgr = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_RGB2BGR)
    return bgr


def draw_font(text, style="楷体", scale=1.0, size=20, c1=(255, 255, 255), c2=(0, 0, 0)):
    """
    绘制汉字
    :param text:
    :param style: 字体风格
    :param scale: 缩放因子
    :param size: 字体大小
    :param c1:字体颜色
    :param c2:背景颜色
    :return:
    """
    image = image_utils.create_image(shape=(size, size, 3), color=c2)
    image = draw_image_text(image, (0, 0), text, style=style, size=size, color=c1)
    if scale < 1.0:
        image = image_utils.get_scale_image(image, scale=scale, offset=(0, 0), color=c2)
    return image


if __name__ == "__main__":
    size = 1024
    point = (0, 0)
    string = "飞"
    image = draw_font(string, style="宋体", size=size, scale=0.8, c2=(0, 255, 0))
    image_utils.cv_show_image("image", image, use_rgb=False)
