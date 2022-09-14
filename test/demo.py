# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""

from pybaseutils import file_utils

if __name__ == "__main__":
    """
    pip install pybaseutils
    """
    file1 = "/home/dm/桌面/image/小学12年级不在1515的有_183.txt"
    file2 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval/class_name2925.txt"
    class_name1 = file_utils.read_data(file1, split=None)
    class_name2 = file_utils.read_data(file2, split=None)
    class_name = [n for n in class_name1 if n not in class_name2]
    # class_name = set(class_name1) - set(class_name2)
    print(len(class_name))
    print(class_name)
