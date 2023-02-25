# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-02-24 17:00:30
    @Brief  : https://zhuanlan.zhihu.com/p/494763838
"""

import tracemalloc
import sys
import cv2
import numpy as np


class Example():
    def __init__(self):
        self.data = []

    def encode(self, frame):
        frame = cv2.imencode('.jpg', frame)[1]
        return frame

    def decode(self, frame):
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # self.data.append(frame.copy())
        return frame

    def decode_append(self, frame):
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        self.data.append(frame.copy())
        return frame

    def crop(self, frame):
        return frame[20:80, 120:920]


class MemoryAnalysis(object):
    def __init__(self):
        """https://zhuanlan.zhihu.com/p/494763838"""
        self._step = 0
        tracemalloc.start()
        self.names = {}

    def step(self, step=1):
        self._step += step

    def start(self):
        current, peak = tracemalloc.get_traced_memory()
        self.start = round(current / 10 ** 6)
        self.last = round(current / 10 ** 6)

    def increase(self, name, *args):
        current, peak = tracemalloc.get_traced_memory()
        excepted = sum([sys.getsizeof(arg) for arg in args])
        excepted = round(excepted / 10 ** 6)
        current = round(current / 10 ** 6)
        increase = current - self.last
        info = f"{self._step:<4} {name:<15} : current memory {current}MB increase {increase}MB; except is {excepted}MB"
        if current == excepted:
            print(info)
        else:
            print(info + "  <<< may be memory leak")
        self.last = current

    def delete_1(self, *args):
        excepted = sum([sys.getsizeof(arg) for arg in args])
        excepted = round(excepted / 10 ** 6)
        self.excepted = -excepted

    def delete_2(self, name):
        current, peak = tracemalloc.get_traced_memory()
        usage = round((current - self.last) / 10 ** 6)
        excepted = self.excepted
        print(f"{self._step:<4} {name:<15} : memory usage chanegd is {usage} MB; except is {excepted} MB")
        self.last = current
        self.excepted = None

    def __del__(self):
        print("__del__")
        tracemalloc.stop()


def demo1():
    # 分析每个部分增加了多少内存
    analysis = MemoryAnalysis()
    analysis.start()
    task = Example()
    for i in range(3):
        analysis.step()
        frame = np.zeros((8000, 8000))
        analysis.increase("imread", frame)
        # ===========================================

        encoded = task.encode(frame)
        analysis.increase("encode", encoded)
        # ===========================================

        analysis.delete_1(frame)
        del frame
        analysis.delete_2("del frame")
        # ===========================================

        frame = task.decode(encoded)
        analysis.increase("decode", frame)
        # ===========================================

        analysis.delete_1(encoded)
        del encoded
        analysis.delete_2("del encoded")
        # ===========================================

        frame2 = task.crop(frame)
        analysis.increase("crop", frame2)
        # ===========================================

        analysis.delete_1(frame)
        del frame
        analysis.delete_2("del frame")
        # ===========================================

        analysis.delete_1(frame2)
        del frame2
        analysis.delete_2("del frame2")


def malloc_utils(tag="", ):
    current, peak = tracemalloc.get_traced_memory()
    print(f"{tag} Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")


def demo2():
    # 分析每个部分增加了多少内存
    task = Example()
    tracemalloc.start()
    for i in range(10):
        frame = np.zeros((8000, 8000))
        malloc_utils(tag="0")
        encoded = task.encode(frame)
        malloc_utils(tag="1")
        frame = task.decode_append(encoded)
        # frame = task.decode(encoded)
        malloc_utils(tag="2")
        frame2 = task.crop(frame)
        print("------------"*10)


if __name__ == "__main__":
    # demo1()
    demo2()
