# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-05-23 11:24:37
    @Brief  : Series是一维数据结构，DataFrame二维表格结构，由多个Series组成（每列是一个Series）
"""
import os

import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, numpy_utils, pandas_utils, json_utils
from pybaseutils.cvutils import corner_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme
from scipy.spatial.distance import cdist
import hashlib
import pandas as pd
import nltk
from rich import print_json

punkt_path = "/home/PKing/nltk_data"
from nltk.tokenize import sent_tokenize
import pprint

import json

data = {
    "name": "张三",
    "age": 30,
    "boxes": [[0, 0, 0, 0], [1, 1, 1, 1], [3, 3, 3, 3]],
    "is_student": False,
    "courses": ["数学", "英语"]
}

# 转换为JSON格式的字符串（带缩进美化）
json_str = json.dumps(data, indent=4, ensure_ascii=False)
print(json_str)
print("----------"*10)
print_json(json.dumps(data, ensure_ascii=False))
