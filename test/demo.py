# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""

from pybaseutils import font_utils, file_utils

font_file = '/home/dm/nasdata/dataset-dmai/handwriting/font-style/zykai-gb2312.ttf'

chars = font_utils.get_font_char(font_file, only_chinese=True)
file = "zykai-gb2312-{}.txt".format(len(chars))
file_utils.write_list_data(file, chars)
