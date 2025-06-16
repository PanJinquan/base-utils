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

if __name__ == '__main__':
    #
    obj_info1 = {'boxes': [[10, 10, 50, 50],
                           [10, 10, 50, 50]],
                 "label": ["A0", "A1"],
                 "info1": {}}
    obj_info2 = {'boxes': [[20, 20, 40, 60],
                           [60, 60, 80, 80],
                           [20, 20, 45, 60],
                           ],
                 "label": ["B0", "B1", "B2"],
                 "info2": {}}
    obj_info2 = image_utils.get_targets(obj_info2, targets=['B2', 'B0'], key='label', keys=['boxes'])
    print(json_utils.formatting(obj_info2))
    print("--------" * 10)
    output = image_utils.get_targets_overlap(obj_info1, obj_info2, iou_th=-1)
    match = output[0]['match']
    match = image_utils.cat_targets(match)
    print(json_utils.formatting(output))
