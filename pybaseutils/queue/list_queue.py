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


class Queue(queue.Queue):
    """普通队列，多个POD请求时，无法实现同步数据"""

    def __init__(self, name="defaultlist", maxsize=10, expire=None):
        """
        queue是一个列表队列，队列大小由maxsize指定
        :param name: 队列名称
        :param maxsize:
        :param expire:
        """
        self.name = name
        self.maxsize = maxsize
        self.expire = expire
        super(Queue, self).__init__(maxsize=maxsize)

    def __del__(self):
        if not self.empty(): self.queue.clear()

    def get_queue(self, ):
        return self.queue

    def set_queue(self, value):
        self.queue = value

    def qsize(self) -> int:
        return super(Queue, self).qsize()

    def size(self) -> int:
        return self.qsize()

    def pop(self, **kwargs):
        """Remove and return an item from the queue,index=0"""
        return super(Queue, self).get(**kwargs)

    def get(self, **kwargs):
        """Remove and return an item from the queue,index=0"""
        return self.pop(**kwargs)

    def put(self, **kwargs):
        """Put an item into the queue,index=n"""
        while self.size() >= self.maxsize: self.get()
        return super(Queue, self).put(**kwargs)


if __name__ == '__main__':
    q = Queue(tag="tag", tid="tid", maxsize=3)
