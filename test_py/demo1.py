# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""

from pybaseutils import file_utils, image_utils

if __name__ == '__main__':
    root = "/home/PKing/Downloads/data1"
    files1 = file_utils.get_files_list_v1(root, postfix=["*.jpg"])
    files2 = file_utils.get_files_list(root, postfix=["*.jpg", "*.png"])
    [print(f) for f in files1]
    print("----" * 10)
    [print(f) for f in files2]
