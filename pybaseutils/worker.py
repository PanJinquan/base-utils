# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-05-01 16:36:18
    @Url    : https://blog.csdn.net/xiaoyu_wu/article/details/102820384
"""

import threading
from queue import Queue
from typing import List
from concurrent.futures import ThreadPoolExecutor


class Base(object):
    def __init__(self, maxsize=5, *args, **kwargs):
        self.input: Queue = None  # 输入队列
        self.output: Queue = Queue(maxsize=maxsize)  # 输出队列

    def set_input(self, q: Queue):
        self.input = q

    def get_input(self):
        return self.input

    def set_output(self, q: Queue):
        self.output = q

    def get_output(self):
        return self.output

    def task(self, *args, **kwargs):
        """定义目标任务"""
        raise NotImplementedError("task not implemented!")

    def start(self, *args, **kwargs):
        """定义启动方法"""
        raise NotImplementedError("start not implemented!")

    def info(self):
        """打印信息"""
        info = ""
        if self.input:
            info += "input size:{}/{} ".format(self.input.qsize(), self.input.maxsize)
        if self.output:
            info += "output size:{}/{} ".format(self.output.qsize(), self.output.maxsize)
        info = "{} {}".format(self.__class__.__name__, info)
        return info


class Worker(Base):
    def __init__(self, maxsize=5, num_worker=2, *args, **kwargs):
        """
        :param input:  输入任务
        :param maxsize: 输出队列最大容量
        :param num_worker:  线程池最大线程数
        """
        super(Worker, self).__init__(maxsize=maxsize, args=args, kwargs=kwargs)
        self.num_worker = num_worker
        self.executor = ThreadPoolExecutor(num_worker)
        self.task_list = []

    def batch(self, batch_size, timeout=2):
        """
        :param batch_size: 获取一个batch的数据
        :param timeout: 超时
        :return: 
        """
        batch = []
        for i in range(batch_size):
            try:
                data = self.input.get(block=True, timeout=timeout)
                batch.append(data)
            except Exception as e:
                # print("{} 等待:{}".format(self.info(), e))
                break
        return batch

    def task(self, *args, **kwargs):
        """定义目标任务"""
        raise NotImplementedError("task not implemented!")

    def _target_v1(self):
        """
        输入和输出顺序一致
        :return:
        """
        while True:
            batch = self.batch(batch_size=self.num_worker)
            for r in self.executor.map(self.task, batch):
                self.output.put(r)

    def _target_v2(self, timeout=1):
        """
        输入和输出顺序不一致，先执行完的线程先输出,性能比_target_v1好
        :param timeout:
        :return:
        """
        while True:
            try:
                data = self.input.get(block=True, timeout=timeout)
                task = self.executor.submit(self.task, data)
                self.task_list.append(task)
            except Exception as e:
                pass
            finally:
                index = []
                for i, task in enumerate(self.task_list):
                    if task.done():
                        self.output.put(task.result(), block=True, timeout=None)
                    else:
                        index.append(i)
                self.task_list = [self.task_list[i] for i in index]

    def target(self):
        return self._target_v2()
        # return self._target_v1()

    def start(self, *args, **kwargs):
        t = threading.Thread(target=self.target, args=args, kwargs=kwargs)
        # 执行线程
        t.start()


class Compose(object):
    def __init__(self, workers: List[Worker]):
        self.workers = workers
        w0 = self.workers[0]
        for w in self.workers[1:]:
            w.set_input(w0.get_output())
            w0 = w

    def start(self, *args, **kwargs):
        for w in self.workers:
            w.start(*args, **kwargs)
