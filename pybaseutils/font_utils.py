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
import PIL
from PIL import Image, ImageDraw, ImageFont
from pybaseutils import image_utils, file_utils
from pybaseutils.font_style import FONT_TYPE, FONT_ROOT
from fontTools.ttLib import TTFont


class FontType(object):
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

    def set_font_type(self, font_file, size):
        self.font_root = os.path.dirname(font_file)
        self.font_style = os.path.basename(font_file)
        self.font_size = size
        self.font_type = ImageFont.truetype(self.font_file, size=size, encoding="utf-8")

    def get_font_type(self):
        return self.font_type


font_type = FontType()


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
    if os.path.isfile(style):
        font = ImageFont.truetype(style, size=size, encoding="utf-8")
    else:
        # simhei.ttf 是字体，你如果没有字体，需要下载
        font_type.set_font_style(style=style, size=size)
        font = font_type.get_font_type()
    draw.text(point, text, fill=color, font=font)
    image = np.asarray(image)
    return image


def draw_font(text, style="楷体", scale=1.0, size=20, c1=(255, 255, 255), c2=(0, 0, 0), center=True):
    """
    绘制汉字
    :param text:
    :param style: 字体风格
    :param scale: 缩放因子
    :param size: 字体大小
    :param c1:字体颜色
    :param c2:背景颜色
    :param center: 是否居中显示
    :return:
    """
    if center:
        image = image_utils.create_image(shape=(size * 2, size * 2, 3), color=c2)
        image = draw_image_text(image, (0, 0), text, style=style, size=size, color=c1)
        if np.sum(image) < 1: return None
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        box = image_utils.get_mask_boundrect_cv(mask, binarize=True, shift=10)
        image = image_utils.get_bbox_crop(image, box)
        image = image_utils.resize_image_padding(image, size=(size, size), color=(0, 0, 0))
    else:
        image = image_utils.create_image(shape=(size, size, 3), color=c2)
        image = draw_image_text(image, (0, 0), text, style=style, size=size, color=c1)
        if np.sum(image) < 1: return None
    if scale < 1.0:
        image = image_utils.get_scale_image(image, scale=scale, offset=(0, 0), color=c2)
    return image


def is_chinese(uchar):
    """
    判断一个字符uchar是否为汉字
    :param uchar:一个字符，如"我"
    :return: True or False
    """
    for ch in uchar:
        if '\u4e00' <= ch <= '\u9fff': return True
    return False


def get_string_chinese(string):
    """
    获得字符串中所有汉字，其他字符删除
    new = re.sub('([^\u4e00-\u9fa5])', '', old) # 字符串删掉除汉字以外的所有字符
    :param string: 
    :return: 
    """
    new = re.sub('([^\u4e00-\u9fa5])', '', string)  # 字符串删掉除汉字以外的所有字符
    return new


def get_string_chinese_number(string):
    """
    获得字符串中所有汉字和数字，其他字符删除
    new = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', old) # 字符串删掉除汉字和数字以外的其他字符
    :param string: 
    :return: 
    """
    new = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', string)  # 字符串删掉除汉字和数字以外的其他字符
    return new


def get_font_char(font_file, only_chinese=False):
    """
    返回字库支持的所有字符
    ord()返回字符对应的ascii码
    chr()返回ascii码对应的字符,python2是unichr
    :param font_file:字库*.ttf文件，如 方正粗黑宋简体.ttf
    :param only_chinese: 是否只返回汉字
    :return:
    """
    font = TTFont(font_file)
    info = font.getBestCmap()
    fonts = []
    for k, v in info.items():
        w = chr(k)
        if only_chinese and not is_chinese(w): continue
        fonts.append(w)
    return fonts


if __name__ == "__main__":
    size = 512
    point = (0, 0)
    string = "我"
    for style, path in FONT_TYPE.items():
        style = "/home/dm/nasdata/dataset-dmai/ziku/1200款精品字体/书法字体库 37款/国祥手写体.ttf"
        # print(style, path)
        image = draw_font(string, style=style, size=size, scale=0.8, c2=(100, 0, 255))
        image_utils.cv_show_image("style", image, use_rgb=False)
