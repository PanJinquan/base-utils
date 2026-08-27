# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils

import cv2

data="ABCDabcd你好我是一名学生1234567890"
fontScale = 1.0
thickness = 1
for text in data:
    text_wh, baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_COMPLEX	, fontScale, abs(thickness))
    print(f"fontScale={fontScale}, thickness={thickness}, text={text}, text_wh={text_wh}, baseline={baseline}")
