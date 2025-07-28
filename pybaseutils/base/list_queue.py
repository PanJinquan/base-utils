# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-15 10:56:37
# @Brief  : 列表队列
# --------------------------------------------------------
"""
import queue


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
        """Remove and return an item from the queue,index=0"""
        return self.queue.get(block=block, timeout=timeout)

    def get(self, index):
        """Remove and return an item from the queue,index=0"""
        return self.queue.queue[index]

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue,index=n"""
        while self.qsize() >= self.maxsize: self.pop()
        return self.queue.put(item, block=block, timeout=timeout)


if __name__ == '__main__':
    q = Queue(maxsize=3)
    q.put(10)
    q.put(11)
    q.put(12)
    q.put(13)
    q.put(14)
    print(q.get_queue())
    print(q.get(0))
    print(q.get_queue())
