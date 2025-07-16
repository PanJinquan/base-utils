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
from pybaseutils import file_utils, image_utils, numpy_utils, pandas_utils, json_utils, text_utils
from pybaseutils.cvutils import corner_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme
from scipy.spatial.distance import cdist
import hashlib
import pandas as pd
import nltk
from rich import print_json
import inspect
from pybaseutils import log
import asyncio
from collections import defaultdict, OrderedDict, namedtuple

if __name__ == "__main__":
    track_history = defaultdict(list) # 创建了一个默认值为空列表的字典
    data = [
        {"B": [0, 0, 0, 0]},
        {"A": [1, 0, 0, 0]},
        {"B": [2, 0, 0, 0]},
        {"A": [3, 0, 0, 0]},
    ]
    for info in data:
        track = track_history[list(info.keys())[0]]
        track.append(list(info.values())[0])  # x, y center point
    print(track_history)
