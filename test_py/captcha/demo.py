# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-05-29 15:27:02
    @Brief  : https://www.jb51.net/python/3199136nj.htm
"""
import random
import numbers
import numpy as np
from captcha.image import ImageCaptcha
from pybaseutils import image_utils

letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letters_lower = "abcdefghijklmnopqrstuvwxyz"
numbers_table = "0123456789"


def random_char(char_nums=(4, 6), upper=True, lower=True, nums=True):
    """
    生成随机字母或数字
    :param length: 
    :param upper: 
    :param lower: 
    :param nums: 
    :return: 
    """
    table = ""
    if upper: table += letters_upper
    if lower: table += letters_lower
    if nums: table += numbers_table
    c = ""
    if isinstance(char_nums, numbers.Number):
        size = (char_nums, char_nums)
    else:
        size = char_nums
    length = int(random.choice(size))
    for i in range(length):
        c += random.choice(table)
    return c


if __name__ == '__main__':
    char_nums = 9  # 字符个数
    size = 48  # 字符大小
    captcha = ImageCaptcha(width=size * char_nums, height=size, font_sizes=(size,))
    for i in range(1000):
        chars = random_char(char_nums=(5, 9))
        image = captcha.generate_image(chars)
        image = np.array(image)
        image_utils.cv_show_image("captcha", image)
