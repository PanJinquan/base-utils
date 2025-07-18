# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-15 10:56:37
# @Brief  : 字典队列
# --------------------------------------------------------
"""
import time
import threading
import traceback
from typing import Any, Dict, List
from collections import defaultdict, OrderedDict, namedtuple

thread_lock = threading.Lock()


class Queue(object):
    """字典队列，key的数据需要手动清理，建议使用TimeQueue，定期清理过期key的缓存"""

    def __init__(self, name="default-queue", maxsize=10, **kwargs):
        """
        queue是一个字典队列，每个key是一个队列，队列大小由maxsize指定
        :param name: 队列名称
        :param maxsize:队列最大长度
        """
        self.name = name
        self.maxsize = maxsize
        self.queue = defaultdict(list)

    def empty(self, ):
        return not self.queue

    def clear(self, key=None):
        keys = list(self.queue.keys()) if key is None else [key]
        for k in keys:
            if k in self.queue: del self.queue[k]

    def __del__(self):
        self.clear()

    def get_queue(self, key):
        return self.queue[key]

    def set_queue(self, key, value):
        self.queue[key] = value

    def qsize(self, key) -> int:
        return len(self.queue[key])

    def pop(self, key, index=0):
        """Remove and return an item from the queue,index=0"""
        return self.queue[key].pop(index)

    def get(self, key, index=0):
        """return an item from the queue,index"""
        return self.queue[key][index]

    def put(self, key, value: Any):
        """Put an item into the queue,index=n"""
        while self.qsize(key) >= self.maxsize:
            self.pop(key, index=0)
        self.queue[key].append(value)
        return self.queue[key]

    def print(self, key=None):
        for k in [key] if key else list(self.queue.keys()):
            print(f"{k}: {self.queue[k]}")
        print("---" * 10)


class TimeQueue(Queue):
    """时序字典队列,每次添加数据都会自动添加时间戳信息，过期的数据会被清理，避免缓存爆炸"""

    def __init__(self, name="time-queue", maxsize=10, sleep=5, expire=30, disp=True):
        """
        queue是一个字典队列，每个key是一个队列，队列大小由maxsize指定
        :param name: 队列名称
        :param maxsize:队列最大长度
        :param sleep: 启动自动清理数据时间间隔，单位秒
        :param expire: 过期时间，超过该时间的数据会被清理，单位秒
        :param disp: 是否定期打印列表数据
        """
        super().__init__(name=name, maxsize=maxsize)
        self.sleep = sleep
        self.expire = expire
        self.disp = disp
        self.strptime = 'strptime'  # 记录添加的数据的时间戳
        # 新建一个时间线程，用于定期清除过期的数据，避免数据累积过大
        self.thread = threading.Thread(target=self.auto_clear, args=(sleep, expire))
        self.thread.daemon = True
        self.thread.start()

    def put(self, key, value: Dict, t=-1):
        """Put an item into the queue,index=n"""
        while self.qsize(key) >= self.maxsize:
            self.pop(key, index=0)
        value[self.strptime] = t if t >= 0 else time.time()  # 增加时间戳信息
        self.queue[key].append(value)
        return self.queue[key]

    def auto_clear(self, sleep=5, expire=1):
        """
        :param sleep: 启动自动清理数据时间间隔，单位秒
        :param expire: 过期时间，超过该时间会被清理，单位秒
        :return:
        """
        while True:
            time.sleep(sleep)
            thread_lock.acquire()
            for key in list(self.queue.keys()):
                if self.qsize(key) == 0:
                    self.clear(key)
                else:
                    t0 = self.get(key, -1)[self.strptime]
                    t1 = time.time()
                    if (t1 - t0) > expire:  self.clear(key)
            if self.disp and not self.empty(): self.print()
            thread_lock.release()


if __name__ == '__main__':
    q = TimeQueue(name="time-queue", maxsize=3, sleep=1, expire=3, disp=True)
    q.put(key="key1", value={"file": "image1.jpg"})
    q.put(key="key1", value={"file": "image2.jpg"})
    q.put(key="key1", value={"file": "image3.jpg"})
    q.put(key="key2", value={"file": "image2.jpg"})
    time.sleep(2)
    q.put(key="key1", value={"file": "image4.jpg"})
    time.sleep(1000)
