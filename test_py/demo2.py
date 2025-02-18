# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import os
import cv2
import random
import types
import torch
import numpy as np
from typing import Callable
from pybaseutils import image_utils, file_utils, text_utils
from pybaseutils.cvutils import video_utils
import cv2
import re

if __name__ == '__main__':
    file = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/action-names.txt"
    outputs = [torch.tensor(1, dtype=torch.int64), torch.tensor(1, dtype=torch.int64)]
    data = torch.stack(outputs)
    print(data)
