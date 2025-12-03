# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-15 10:56:37
# @Brief  : 列表队列
# --------------------------------------------------------
"""
import math
import queue
import time


class Queue():
    """普通队列，多个POD请求时，无法实现同步数据"""

    def __init__(self, name="defaultlist", maxsize=10, **kwargs):
        """
        queue是一个列表队列，队列大小由maxsize指定
        :param name: 队列名称
        :param maxsize:
        """
        self.name = name
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)

    def empty(self, ):
        return self.queue.empty()

    def __del__(self):
        if not self.queue.empty(): self.queue.queue.clear()

    def get_queue(self, ):
        return self.queue.queue

    def set_queue(self, queue):
        self.queue = queue

    def qsize(self) -> int:
        return self.queue.qsize()

    def pop(self, block=True, timeout=None):
        """
        Remove and return an item from the queue,index=0
        :param block: 是否阻塞等待
        :param timeout: 超时时间
        :return: 弹出的队列数据
        """
        try:
            out = self.queue.get(block=block, timeout=timeout)
        except:
            out = None
        return out

    def pop_items(self, nums=1, block=True, timeout=None):
        """
        弹出队列的多个项
        :param nums: 弹出队列的数量
        :param block: 是否阻塞等待
        :param timeout: 超时时间
        :return: 弹出的队列数据
        """
        items = []
        for i in range(nums):
            item = self.pop(block=block, timeout=timeout)
            if item: items.append(item)
        return items

    def get(self, index=0):
        """get an item from the queue,index=0"""
        return self.queue.queue[index]

    def get_items(self, nums=1, index: list = []):
        """
        获取队列的多个项
        :param nums: 获取队列的数量
        :param index: 获取队列的索引
        :return: 获取的队列数据
        """
        if index:
            items = [self.queue.queue[i] for i in index]
        else:
            items = [self.queue.queue[i] for i in range(nums)]
        return items

    def put(self, item, block=False, timeout=None):
        """
        Put an item into the queue,index=n
        :param item: 要放入队列的项
        :param block: True当队列满了,阻塞等待插入数据; False,不等待,直接弹出队头数据再插入数据
        :param timeout: 超时时间
        :return:
        """
        while self.qsize() >= self.maxsize and not block: self.pop()
        return self.queue.put(item, block=block, timeout=timeout)

    def get_window(self, winsize=1, overlap=0.0, block=True, timeout=None):
        """
        获取队列的窗口数据
        :param winsize: 窗口大小
        :param overlap: 窗口重叠率
        :param block: 是否阻塞等待
        :param timeout: 超时时间
        :return: 队列窗口数据
        """
        while self.qsize() < winsize and block:  # 等待队列数据足够
            time.sleep(0.05)
        if self.qsize() >= winsize:
            size = int(winsize * overlap)
            data1 = self.pop_items(nums=winsize - size, block=block, timeout=timeout)
            data2 = self.get_items(nums=size)
            return data1 + data2
        return []


if __name__ == '__main__':
    q = Queue(maxsize=3)
    q.put({"file": "1.jpg"})
    q.put({"file": "2.jpg"})
    q.put({"file": "3.jpg"})
    q.put({"file": "4.jpg"})
    q.put({"file": "5.jpg"})
    print(q.get_queue())
    print(q.get(0))
    print(q.get_queue())
    print(q.pop(3))
    print(q.get_queue())
    print(1 / 2)
