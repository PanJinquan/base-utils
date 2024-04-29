# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-05-08 11:25:41
"""

import numpy as np


class QueueTable(object):
    """
    队列表中，每一行是一条项(Item),每一列是一个序列(Sequence)
    """

    def __init__(self, win_size, pad=True):
        """
        :param win_size: 队列大小
        :param pad: True : 如果table不足win_size长度时，会进行填充
                        False: 不会进行填充
        """
        # 窗口大小
        self.win_size = win_size
        self.padding = pad
        # 数据队列表
        self.data = []
        self.clear()

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "".join(self.data)

    def get_data(self):
        return self.data

    def get_size(self):
        return self.__len__()

    def _padding(self, row=-1):
        """用最新的数据信息填充"""
        for i in range(self.win_size - len(self)):
            item = self.get_item(row)
            self.data.append(item)

    def set_element(self, e, row=-1, col=0):
        """
        修改table中元素值,
        :param e:
        :param row:默认-1 最后(最新的)一行数据
        :param col:默认第一列数据
        :return:
        """
        self.data[row][col] = e

    def put_element(self, e, col=0):
        """
        添加元素到队列表中，其余元素会复制最新值
        :param e:
        :param col:默认第一列数据
        :return:
        """
        item = self.data[-1].copy()
        item[col] = e
        self.put_item(item)

    def put_item(self, item):
        """
        添加item(行)到队列表中
        :param item:
        :return:
        """
        if len(self.data) == self.win_size:
            self.data.pop(0)
        self.data.append(item)
        if self.padding:
            self._padding()

    def get_item(self, row=-1):
        """
        获取队列表中第index的item(行)数据
        :param row:
        :return:
        """
        return self.data[row]

    def push_seq(self, seq=0):
        pass

    def get_seq(self, col=0):
        """
        获得队列表中时序seq(列)数据,比如某个关键点的时序坐标信息
        :param col:
        :return:
        """
        seq = []
        for row in range(len(self)):
            item = self.get_item(row)
            seq.append(item[col])
        return seq

    def clear(self):
        """清空所有数据"""
        self.data.clear()
        self.data = []


class TimeAlignment(object):
    def __init__(self, win_size=10):
        self.queue = QueueTable(win_size=win_size)

    def push(self, data):
        if len(data) > 0:
            self.queue.put_item(data)


if __name__ == "__main__":
    win_size = 10
    q = QueueTable(win_size, pad=True)
    data0 = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]
    data1 = [(0, 0, 0, 0), (11, 11, 11, 11), (12, 12, 12, 12)]
    data2 = [(0, 0, 0, 0), (21, 21, 21, 21), (22, 22, 22, 22)]
    data3 = [(0, 0, 0, 0), (31, 31, 31, 31), (32, 32, 32, 32)]
    q.put_item(data0)
    q.put_item(data1)
    q.put_item(data2)
    q.put_item(data3)
    q.set_element((999, 999, 999, 999))
    q.put_element((1000, 1000, 1000, 1000))
    t = q.get_seq(col=1)
