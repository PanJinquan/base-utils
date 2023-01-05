# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-05-01 16:36:18
    @Url    : https://blog.csdn.net/xiaoyu_wu/article/details/102820384
            python多进程&多线程
            由于多线程受GIL全局解释器锁的影响，多进程比多线程性能好，
            python多线程采用GIL(全局解释器锁)加锁机制，线程在执行代码时，必须先获得这把锁，才获得CPU执行代码指令。
            如果这把锁被其他线程占用，该线程就只能等待，等到占有该锁的线程释放锁。在进行数值计算时，python的多线程可能会更慢；
            但在进行网络爬虫，文件读写的时候，使用多线程会更快（因为每个线程执行的时候都是在请求外部的资源，而非CPU内部的计算
            参考：https://blog.csdn.net/weixin_42176112/article/details/117790945
"""

import threading
import time
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

# 创建线程锁
thread_lock = threading.Lock()


def thread_safety(func, *args, **kwargs):
    """
    使用以下加锁方式
    with 锁对象：
        do something

    和以下方式一样的效果
    lock.acquire()
    try:
        do something
    finnaly:
        lock.release()
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    with thread_lock:
        r = func(*args, **kwargs)
    return r


def consumer(image_path: str):
    """
    :param image_path:
    :return:
    """
    t = int(image_path.split(".")[0])
    time.sleep(t)
    # with thread_lock:
    print("正在处理数据：{}  ".format(image_path))
    return image_path


class ProcessPool(object):
    """进程池"""

    def __init__(self, max_workers=2):
        """
        :param max_workers:  进程池最大线程数
        """
        self.pool = Pool(processes=max_workers)

    def task_map(self, func: Callable, inputs: List):
        """进程任务，返回结果有序"""
        result = []
        for r in self.pool.map(func, inputs):
            result.append(r)
        return result

    def task_apply_async(self, func: Callable, inputs: List):
        """进程任务，返回结果有序"""
        result = [self.pool.apply_async(func, args=(i,)) for i in inputs]
        result = [r.get() for r in result]
        return result

    def close(self):
        self.pool.close()

    def __getstate__(self):
        """Fix a bug:NotImplementedError: pool objects cannot be passed between processes or pickled"""
        self_dict = self.__dict__.copy()
        del self_dict['pool']  # 删除self.pool的名称
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class ThreadPool(object):
    """线程池"""

    def __init__(self, max_workers=2):
        """
        :param max_workers: 线程池最大线程数
        """
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func: Callable, *args, **kwargs):
        """递交线程任务"""
        t = self.pool.submit(func, *args, **kwargs)
        return t

    def task_map(self, func: Callable, inputs: List):
        """线程任务，返回结果有序(map与submit的性能基本一致)"""
        # 通过executor的 map 获取已经完成的task的值
        result = []
        for r in self.pool.map(func, inputs):
            result.append(r)
        return result

    def task_submit(self, func: Callable, inputs: List):
        """线程任务，返回结果无序(map与submit的性能基本一致)"""
        task_list = [self.pool.submit(func, p) for p in inputs]
        result = []
        for task in as_completed(task_list):
            result.append(task.result())
        return result

    def task_submit_v1(self, func: Callable, inputs: List):
        """线程任务，返回结果无序"""
        task_list = [self.pool.submit(func, p) for p in inputs]
        result = []
        while len(task_list) > 0:
            index = []
            for i, task in enumerate(task_list):
                if task.done():
                    result.append(task.result())
                else:
                    index.append(i)
            task_list = [task_list[i] for i in index]
        return result

    def shutdown(self, wait=True):
        self.pool.shutdown(wait=wait)


def thread_lock_decorator():
    def decorator(func):
        def wrapper(*args, **kwargs):
            with thread_lock:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def performanceThreadPool():
    from pybaseutils import time_utils
    tp = ThreadPool(max_workers=4)
    contents = ["1.jpg", "4.jpg", "4.jpg", "4.jpg", "2.jpg"]
    print(contents)
    with time_utils.Performance("task_map") as p:
        result1 = tp.task_map(func=consumer, inputs=contents)
    with time_utils.Performance("task_submit") as p:
        result2 = tp.task_submit(func=consumer, inputs=contents)
    print(result1)
    print(result2)


def performanceProcessPool():
    from pybaseutils import time_utils
    tp = ProcessPool(max_workers=4)
    # contents = ["1.jpg", "4.jpg", "4.jpg", "4.jpg", "2.jpg"]
    contents = ["1.jpg", "5.jpg", "4.jpg", "3.jpg", "2.jpg"]
    print(contents)
    with time_utils.Performance("task_map") as p:
        result1 = tp.task_map(func=consumer, inputs=contents)
    with time_utils.Performance("task_apply_async") as p:
        result2 = tp.task_apply_async(func=consumer, inputs=contents)
    print(result1)
    print(result2)


if __name__ == "__main__":
    # performanceThreadPool()
    performanceProcessPool()
