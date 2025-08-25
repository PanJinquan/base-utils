# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-22 10:40:45
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
from pybaseutils import json_utils, image_utils
from pybaseutils.dataloader import parser_labelme

if __name__ == "__main__":
    data =[5,1,1,5,16]
    outs = np.median(data)
    print(outs)
