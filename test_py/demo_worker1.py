# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-04-25 16:20:17
    @Brief  :
"""
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable
from basetrainer.engine import trainer
threadLock = threading.Lock()


# threadLock.acquire()  # 加锁
# threadLock.release()  # 释放锁

class Producer(queue.Queue):
    def __init__(self, maxsize=4):
        super(Producer, self).__init__(maxsize=maxsize)

    def task(self, nums):
        """循环发送数据"""
        done = True
        count = 0
        while done:
            time.sleep(0.5)  # 每隔一段时间发送一次数据
            data = "{}.jpg".format(count)
            self.put(data)
            done = nums > count
            count += 1

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self.info(tag="put")

    def get(self, block=True, timeout=None):
        self.info(tag="get")
        data = super().get(block=block, timeout=timeout)
        return data

    def full(self):
        return super().full()

    def info(self, tag):
        print("tag:{},size:{}/{}".format(tag, self.qsize(), self.maxsize))

    def test(self, nums=10):
        p = threading.Thread(target=self.task, args=(nums,))
        # 执行线程
        p.start()


class Worker(object):
    def __init__(self, max_workers=2, maxsize=10):
        """
        :param max_workers:  线程池最大线程数
        :param maxsize:
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers)

    def submit(self, func: Callable, *args, **kwargs):
        """递交线程任务"""
        t = self.executor.submit(func, *args, **kwargs)
        return t

    def get(self, producer: Producer, nums):
        done = True
        count = 0
        data_list = []
        timeout = 2
        while done:
            try:
                data = producer.get(block=True, timeout=timeout)
                data_list.append(data)
            except Exception as e:
                print("等待:{}".format(e))
            finally:
                done = nums < count
                count += 1
        return data_list

    def task(self, data):
        with threadLock:
            print("processing data:{}".format(data))
            time.sleep(1)
        return data

    def test(self, producer: Producer):
        """线程任务"""
        # 通过executor的 map 获取已经完成的task的值
        done = True
        while done:
            data_list = self.get(producer, nums=self.max_workers)
            for r in self.executor.map(self.task, data_list):
                print("result:{}".format(r))


if __name__ == "__main__":
    p = Producer()
    p.test()
    print("create Producer")
    w = Worker(p)
    w.test()
