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
from pybaseutils import image_utils, file_utils
import cv2


class RedisQueue(object):
    """Redis队列，多个POD请求时，通过Redis实现同步数据"""

    def __init__(self, tag, tid, maxsize):
        self.tag = tag
        self.tid = tid
        self.maxsize = maxsize
        self.key = f"{self.tag}-{self.tid}"
        # self._queue = [["A"], ["B"]]

    @property
    def queue(self):
        pass

    @queue.setter
    def queue(self, value):
        self._queue = value
        print(self._queue)


data = RedisQueue(tag="", tid="", maxsize="")
print(data.queue)
data.queue = "OK2"
print(data.queue)
