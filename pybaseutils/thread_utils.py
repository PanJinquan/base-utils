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
from concurrent.futures import ThreadPoolExecutor

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


def consumer(image_path):
    """
    :param image_path:
    :return:
    """
    time.sleep(1)
    with thread_lock:
        print("正在处理数据：{}  ".format(image_path))
    return image_path


class ThreadPool(ThreadPoolExecutor):
    def __init__(self, max_workers=2, maxsize=10):
        """
        :param max_workers:  线程池最大线程数
        :param maxsize:
        """
        self.executor = ThreadPoolExecutor(max_workers)

    def submit(self, func: Callable, *args, **kwargs):
        """递交线程任务"""
        t = self.executor.submit(func, *args, **kwargs)
        return t

    def task(self, func: Callable, inputs: List):
        """线程任务"""
        # 通过executor的 map 获取已经完成的task的值
        result = []
        for r in self.executor.map(func, inputs):
            result.append(r)
        return result


def thread_lock_decorator():
    def decorator(func):
        def wrapper(*args, **kwargs):
            with thread_lock:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    tp = ThreadPool(max_workers=2)
    contents = ["{}.jpg".format(i) for i in range(10)]
    print(contents)
    result = tp.task(func=consumer, inputs=contents)
    print("result:{}".format(result))
