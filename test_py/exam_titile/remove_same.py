# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-27 20:12:32
    @Brief  :
"""
from pybaseutils import file_utils, image_utils

if __name__ == '__main__':
    file = ""
    data = file_utils.read_data(file)
    data1 = []
    for d in data:
        data1 += d
    data2 = list(set(data1))
    data2 = sorted(data2)
    for d in data2:
        print(d)
