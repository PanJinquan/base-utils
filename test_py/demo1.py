# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-22 10:40:45
# @Brief  :
# --------------------------------------------------------
"""
import time
import numpy as np
import re
from pybaseutils import text_utils


def date2stamp(date, format='%Y-%m-%d %H:%M:%S') -> float:
    """将日期格式转换为时间戳"""
    try:
        stamp = time.mktime(time.strptime(date, format))
    except:
        stamp = -1
    return stamp


# 示例用法
if __name__ == "__main__":
    format = '%Y%m%d %H%M%S'  # 年月日 时分秒(中间有一个空格)
    data_list = ['2025-05-07&14:28:22']
    ymd_len = 8
    for data in data_list:
        video_date = text_utils.find_digits(data)
        video_date = "".join(video_date)
        video_date = text_utils.insert_string(video_date, ymd_len, sub=" ")
        video_time = date2stamp(video_date,format)
        print(video_time,video_date)
