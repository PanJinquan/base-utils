# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-25 17:42:55
    @Brief  :
"""
import os
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


def rename_files(src, dst, prefix="", postfix=None, remove=False):
    """
    对当前目录的文件进行重命名
    :param src: 原始目录
    :param dst: 输出目录
    :param prefix: 重命名前缀
    :param remove: 是否删除原始文件
    :return:
    """
    file_utils.rename_files(src, dst, prefix=prefix, postfix=postfix, remove=remove)


if __name__ == '__main__':
    image_dir = "/home/PKing/edudata/dataset/AIJE/【TOP】技能人才系统_数据集管理/05-东莞基地/东莞基地-2025-08-19/images"
    output = image_dir
    rename_files(image_dir, output, prefix="dg_image", remove=True)
