# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-07 17:40:27
"""
import os

# windows: C:\Windows\Fonts可查看系统支持的字体
# Linux  : /usr/share/fonts/truetype可查看系统支持的字体

FONT_ROOT = os.path.dirname(__file__)
FONT_TYPE = {
    "楷体": "simkai.ttf",
    "宋体": "simsun.ttc",
    "仿宋": "simfang.ttf",
    "黑体": "simhei.ttf",
    # "simhei": "simhei.ttf", # 与黑体相近
    # "等线": "Deng.ttf",
    "方正": "方正粗黑宋简体.ttf",
}
