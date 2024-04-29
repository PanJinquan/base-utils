# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-01-19 19:29:47
"""
import numpy as np
from pybaseutils.filter import QueueTable


class MotionFilter():
    def __init__(self, win_size, decay=0.9):
        """
        :param win_size:记录历史信息的窗口大小
        :param decay: 衰减系数，值越大，历史影响衰减的越快，平滑力度越小
        """
        self.decay = decay
        self.last = None
        self.curr = None
        print("prob:{}".format(self.decay))

    def update(self, point):
        if point[0] > 0 and point[1] > 0:
            self.curr = point
            if self.last is None: self.last = point

    def predict(self):
        if isinstance(self.last, np.ndarray):
            point = self.filter()
        else:
            point = np.array([0, 0])
        return point

    def filter(self, ):
        curr = self.curr * self.decay + (1 - self.decay) * self.last
        self.last = curr
        return curr

    @staticmethod
    def get_weight(n, decay=0.5):
        """
        当n=5,decay=0.5时，对应的衰减权重为，越远的权重越小
        w=[0.0625 0.0625 0.125  0.25   0.5   ]
        :param n:
        :param decay: 衰减系数，值越大，历史影响衰减的越快，平滑力度越小
        :return:
        """
        r = decay / (1 - decay)
        # 计算衰减权重
        w = [1]
        for i in range(1, n):
            w.append(sum(w) * r)
        # 进行归一化
        w = w / np.sum(w)
        return w
