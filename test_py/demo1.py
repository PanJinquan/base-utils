# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import threading
import time
import torch
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool


def torch_version_id(v: str):
    vid = v.split(".")
    vid = float("{}.{:0=2d}".format(vid[0], int(vid[1])))
    return vid


def get_torch_version():
    try:
        v = torch.__version__
        print("torch.version:{}".format(v))
        vid = torch_version_id(v)
    except Exception as e:
        vid = None
    return vid


if __name__ == "__main__":
    print(320 * 78 + 3600.00 * 46 + 120.00 * 12)
