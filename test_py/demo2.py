# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import random
import types
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re

if __name__ == '__main__':
    # 示例使用
    subs = ["拿", '手*']  #
    texts = ["拿着", "手拿安全帽", "手拿绝缘手套", "手拿工具袋"]
    # 使用通配符查找子串
    matches = text_utils.find_match_texts(texts, subs, org=True)
    print(matches)
