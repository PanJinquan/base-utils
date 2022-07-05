# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-05-25 18:19:34
    @Brief  :
"""
import time
import numpy as np
from typing import Dict, List


class Recorder(object):
    def __init__(self):
        self.content: Dict = {}

    def push(self, key, v):
        if key in self.content:
            self.content[key]["total"] = self.content[key]["total"] + v
            self.content[key]["count"] = self.content[key]["count"] + 1
            self.content[key]["avg"] = self.content[key]["total"] / self.content[key]["count"]
        else:
            self.content[key] = {"avg": v, "total": v, "count": 1}

    def get(self, key) -> Dict:
        return self.content[key]

    def reset(self):
        self.content: Dict = {}

    def info(self, key):
        print(self.get(key))


recorder = Recorder()


def performance(tag=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # torch.cuda.synchronize()
            t0 = time.time()
            result = func(*args, **kwargs)
            # torch.cuda.synchronize()
            t1 = time.time()
            key = tag if tag else str(func.__name__)
            recorder.push(key=key, v=(t1 - t0) * 1000)
            content = recorder.get(key)
            elapsed = "avg:{:.5f}ms\t total:{:.5f}ms\t count:{}".format(content["avg"],
                                                                        content["total"],
                                                                        content["count"])
            print("{} call {} elapsed: {}".format(tag, func.__name__, elapsed))
            return result

        return wrapper

    return decorator


class Performance(object):
    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # 参数为异常信息
        self.t1 = time.time()
        recorder.push(key=self.tag, v=(self.t1 - self.t0) * 1000)
        self.info(key=self.tag)

    def info(self, key):
        content = recorder.get(key)
        elapsed = "avg:{:.5f}ms\t total:{:.5f}ms\t count:{}".format(content["avg"],
                                                                    content["total"],
                                                                    content["count"])
        print("{} elapsed: {}".format(self.tag, elapsed))

    def task(self):
        pass


@performance("")
def targe_func1():
    time.sleep(1)


@performance("")
def targe_func2():
    time.sleep(0.5)


def targe_func3():
    with Performance("targe_func3") as p:
        time.sleep(1)


def targe_func4():
    with Performance("targe_func4") as p:
        time.sleep(0.5)


def targe_func():
    targe_func1()
    targe_func2()
    targe_func3()
    targe_func4()


if __name__ == '__main__':
    for i in range(10):
        targe_func()
