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
from pycocotools import cocoeval
from pybaseutils import image_utils, file_utils
from pybaseutils.singleton_utils import Singleton
import cv2

import threading
import time

count = 0

import threading
import time

import threading

count=0
class AssetsMethod(metaclass=Singleton):
    def __init__(self):
        self.run_threading()

    def task1(self, interval=1):
        global count
        while True:
            print("task1----------------", f"[{count}]")
            time.sleep(interval)

    def task2(self, interval=1):
        global count
        while True:
            count += 1
            print("task2----------------", f"[{count}]")
            time.sleep(interval)

    def run_threading(self):
        thread1 = threading.Thread(target=self.task1, name="task1")
        thread2 = threading.Thread(target=self.task2, name="task2")
        thread1.start()
        thread2.start()


if __name__ == '__main__':
    assets = AssetsMethod()
    print(AssetsMethod())
    print(AssetsMethod())
    print(AssetsMethod())
