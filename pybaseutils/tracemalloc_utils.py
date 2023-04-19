# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-05-25 18:19:34
    @Brief  : https://zhuanlan.zhihu.com/p/494763838
    @Brief  : https://zhuanlan.zhihu.com/p/38600861(不要在Python 3中缓存Exception对象)
"""
import cv2
import copy
import numpy as np
import tracemalloc
import time
from typing import Dict, List

IGNORE = ["lib/python", "plugins/python/helpers"]
MACHER = []


class MemoryAnalysis(object):
    """内存分析工具"""

    def __init__(self, keys=MACHER, ignore=IGNORE, mem_diff=0):
        """
        :param keys: 需要匹配相关的字段内容
        :param ignore: 需要忽略不相关的字段内容
        :param mem_diff: 过滤内存存在变化的字段(单位B)
        """
        tracemalloc.start()
        self.count = 0
        self.keys = list(set(keys)) if keys else MACHER
        self.ignore = list(set(ignore)) if ignore else IGNORE
        self.mem_diff = mem_diff
        self.snapshot1 = None  # 第一次快照
        self.snapshot2 = None  # 当前快照
        self.t1 = time.time()
        self.t2 = time.time()
        self.dt = 0
        self.start_c = 0
        self.start_t = 0.1

    def start(self, start_c=0, start_t=0.1):
        """
        设置开始统计次数和时间，当start_c和start_t同时满足时间，开始统计内存变化
        :param start_c: 开始次数
        :param start_t: 开始时间间隔(单位秒)
        :return:
        """
        self.start_c = start_c
        self.start_t = start_t

    def take_snapshot(self):
        return tracemalloc.take_snapshot()

    def add_snapshot(self):
        """
        记录第一次和最后一次的内存快照，第一次条件由(start_c,start_t)指定
        :return:
        """
        self.dt = (time.time() - self.t1)
        if self.count <= self.start_c or self.dt <= self.start_t:
            self.snapshot1 = tracemalloc.take_snapshot()
        self.snapshot2 = tracemalloc.take_snapshot()
        self.count += 1
        return self.snapshot2

    def get_stats(self, snapshot1=None, snapshot2=None, topk=-1, key_type='lineno', cumulative=False):
        """
        :param snapshot1:
        :param snapshot2:
        :param topk:
        :param key_type: key_type is ('traceback', 'filename', 'lineno')
        :param cumulative:
        :return:
        """
        if snapshot1 is None and snapshot2 is None and \
                (self.count <= self.start_c or self.dt <= self.start_t): return
        if not snapshot1: snapshot1 = self.snapshot1
        if not snapshot2: snapshot2 = self.snapshot2
        stats = snapshot2.compare_to(snapshot1, key_type=key_type, cumulative=cumulative)  # 快照对比
        stats = stats[:min(len(stats), topk)] if topk > 0 else stats
        stats = self.ignorer(stats, ignore=self.ignore)
        stats = self.matcher(stats, keys=self.keys)
        stats = [s for s in stats if abs(s.size_diff) > self.mem_diff]
        self.print_info(stats)

    @staticmethod
    def ignorer(stats, ignore=[]):
        """忽略不相关的字段"""
        if not ignore: return stats
        out1 = []
        for s in stats:
            have = False
            for k in ignore:
                if k in str(s.traceback):
                    have = True
                    break
            if not have: out1.append(s)
        return out1

    @staticmethod
    def matcher(stats, keys=[]):
        """匹配相关的字段"""
        if not keys: return stats
        out1 = []
        for s in stats:
            have = False
            for k in keys:
                if k in str(s.traceback):
                    have = True
                    break
            if have: out1.append(s)
        return out1

    def print_info(self, stats):
        for stat in stats:
            print(f"{stat}\t")


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


def demo2():
    # 分析每个部分增加了多少内存
    task = Example()
    memory = MemoryAnalysis()
    memory.start(start_c=0, start_t=10)
    for i in range(100):
        s1 = memory.add_snapshot()
        frame = np.zeros((8000, 8000))
        encoded = task.encode(frame)
        frame = task.decode_append(encoded)
        # frame = task.decode(encoded)
        frame2 = task.crop(frame)
        s2 = memory.add_snapshot()
        # memory.get_stats(s1, s2, topk=-1)
        memory.get_stats(topk=-1)
        print(i, "------------" * 10)


if __name__ == "__main__":
    # demo1()
    demo2()
