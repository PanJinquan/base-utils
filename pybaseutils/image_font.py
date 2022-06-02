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
from pybaseutils.font_style import FONT_TYPE, FONT_ROOT


class FontStyle(object):
    def __init__(self, root=FONT_ROOT, style="宋体", size=20):
        """
        :param root: 安装字体的路径
        :param style: 选择字体风格
        :param size: 旋转字体大小
        """
        self.font_root = root
        self.font_style = style
        self.font_size = size
        self.font_file = os.path.join(self.font_root, FONT_TYPE[style])
        self.font_type = None
        if os.path.exists(self.font_file):
            self.font_type = ImageFont.truetype(self.font_file, size=self.font_size, encoding="utf-8")

    def set_root(self, root):
        """设置字体的路径"""
        self.__init__(root=root, style=self.font_style, size=self.font_size)

    def set_font_style(self, style, size):
        """设置字体风格和大小"""
        self.__init__(root=self.font_root, style=style, size=size)

    def get_font_style(self):
        return self.font_type


font_style = FontStyle()


def draw_image_text(image, point, text, style="楷体", size=20, color=(255, 255, 255)):
    """
    在图像中显示汉字
    https://blog.csdn.net/weixin_44237337/article/details/119817801
    :param image: RGB字体背景
    :param point:
    :param text:
    :param style: 字体风格
    :param size:
    :param color:
    :return:
    """
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    # font = ImageFont.truetype("simhei.ttf", size=size, encoding="utf-8")
    font_style.set_font_style(style=style, size=size)
    font_type = font_style.get_font_style()
    draw.text(point, text, fill=color, font=font_type)
    image = np.asarray(image)
    return image


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
    size = 512
    point = (0, 0)
    string = "你"
    for style, path in FONT_TYPE.items():
        print(style, path)
        image = draw_font(string, style=style, size=size, scale=0.8, c2=(100, 0, 255))
        image_utils.cv_show_image(style, image, use_rgb=False)
