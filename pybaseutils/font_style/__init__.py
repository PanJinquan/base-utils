# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-05-07 17:40:27
"""
import os

# TODO 如果字体缺失，可以把字体拷贝到系统字体目录：cp -r font_style/*  ~/.fonts
# windows: C:\Windows\Fonts可查看系统支持的字体
# Linux  : /usr/share/fonts/truetype  、 ~/.fonts 或者~/.local/share/fonts中 可查看系统支持的字体

BASE_FONT = os.path.dirname(__file__)

# 可能的字体路径列表
FONT_ROOT = [
    os.path.expanduser("~/.local/share/fonts"),
    os.path.expanduser("~/.local/share/truetype"),
    os.path.expanduser("~/.fonts"),  # 用户字体目录
    BASE_FONT,  # 用户字体目录
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts/opentype",
]


def get_all_files(file_dir, postfix=[]):
    """获取file_dir目录下，所有文本路径，包括子目录文件"""
    file_list = []
    for walk in os.walk(file_dir):
        # paths = [os.path.join(walk[0], file).replace("\\", "/") for file in walk[2]]
        paths = [os.path.join(walk[0], file) for file in walk[2]]
        file_list.extend(paths)
    if not file_list: return file_list
    out_list = []
    for file in file_list:
        p = f"*.{os.path.basename(file).split('.')[-1]}"
        if p in postfix: out_list.append(file)
    return out_list


def get_system_fonts(root=[]):
    """
    获得字体路径
    默认字体：ImageFont.load_default()
    :param root: 字体路径
    :return:
    """
    if root and isinstance(root, str): root = [root]
    font_root = root if root else FONT_ROOT
    font_tables = {
        "楷体": os.path.join(BASE_FONT, "simkai.ttf"),
        "simkai": os.path.join(BASE_FONT, "simkai.ttf"),
        "宋体": os.path.join(BASE_FONT, "simsun.ttc"),
        "simsun": os.path.join(BASE_FONT, "simsun.ttc"),
        "仿宋": os.path.join(BASE_FONT, "simfang.ttf"),
        "simfang": os.path.join(BASE_FONT, "simfang.ttf"),
        "黑体": os.path.join(BASE_FONT, "simhei.ttf"),
        "simhei": os.path.join(BASE_FONT, "simhei.ttf"),
        "方正": os.path.join(BASE_FONT, "方正粗黑宋简体.ttf"),
        # "等线": os.path.join(FONT_ROOT, "Deng.ttf"),
    }
    for dir in font_root:
        files = get_all_files(dir, postfix=['*.ttf', '*.ttc', '*.otf'])
        for file in files:
            name = os.path.basename(file).split(".")[0].lower()
            font_tables[name] = file
    for name in list(font_tables.keys()):
        if not os.path.exists(font_tables[name]): font_tables.pop(name)
    return font_tables


FONT_TABLES = get_system_fonts()
