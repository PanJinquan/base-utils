# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import re
import random
import types
import numpy as np
import time
import datetime
import threading
import torch
from pycocotools import cocoeval
from pybaseutils import image_utils, file_utils
from pybaseutils.singleton_utils import Singleton

if __name__ == '__main__':
    data = [[1, 2, 3, 4, 5, 6], [11, 21, 0.3, 0.4, 0.5, 0.6]]
    age_pred1 = torch.Tensor(data)
    out1 = torch.nn.functional.normalize(age_pred1, p=1, dim=1)
    print(out1)
    age_pred2 = np.asarray(data)
    out2 = age_pred2 / np.reshape(np.sum(np.abs(age_pred2), axis=1), newshape=(-1, 1))
    print(out2)
