# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-05-01 16:36:18
    @Url    : https://blog.csdn.net/xiaoyu_wu/article/details/102820384
"""

import threading
import time
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class ThreadPool(object):
    def __init__(self, max_workers=2):
        """
        :param max_workers:  线程池最大线程数
        :param maxsize:
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func: Callable, *args, **kwargs):
        """递交线程任务"""
        t = self.executor.submit(func, *args, **kwargs)
        return t

    def task_map(self, func: Callable, inputs: List):
        """线程任务，返回结果有序(map与submit的性能基本一致)"""
        # 通过executor的 map 获取已经完成的task的值
        result = []
        for r in self.executor.map(func, inputs):
            result.append(r)
        return result

    def task_submit(self, func: Callable, inputs: List):
        """线程任务，返回结果无序(map与submit的性能基本一致)"""
        task_list = [self.executor.submit(func, p) for p in inputs]
        result = []
        for task in as_completed(task_list):
            result.append(task.result())
        return result

    def task_submit_v1(self, func: Callable, inputs: List):
        """线程任务，返回结果无序"""
        task_list = [self.executor.submit(func, p) for p in inputs]
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
        self.executor.shutdown(wait=wait)


def thread_lock_decorator():
    def decorator(func):
        def wrapper(*args, **kwargs):
            with thread_lock:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    import random
    from pybaseutils import debug

    tp = ThreadPool(max_workers=4)
    # contents = ["{}.jpg".format(i / 5) for i in range(10)]
    # random.shuffle(contents)
    contents = ["1.jpg", "4.jpg", "4.jpg", "4.jpg", "2.jpg"]
    print(contents)
    t0 = debug.TIME()
    result1 = tp.task_map(func=consumer, inputs=contents)
    t1 = debug.TIME()
    print()
    result2 = tp.task_submit(func=consumer, inputs=contents)
    t2 = debug.TIME()
    print("task_map   :{}".format(debug.RUN_TIME(t1 - t0)))
    print("task_submit:{}".format(debug.RUN_TIME(t2 - t1)))
    print(result1)
    print(result2)
