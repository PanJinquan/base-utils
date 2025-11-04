# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import time
from datetime import datetime
import numpy as np
from typing import Dict, List


def get_time(format="p"):
    """
    :param format:
    :return:
    """
    if format.lower() == "s":  # 精确到秒
        # time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        time = datetime.now().strftime("%Y%m%d%H%M%S")
    elif format.lower() == "p":  # 精确到微妙
        # time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')  # 20200508_143059_379116
        time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        time = time[:-2]
    elif format.lower() == "y":  # 2025-06-18 11:02:05
        # time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")  #
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        time = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
    return time

def date2stamp(date, format='%Y-%m-%d %H:%M:%S') -> float:
    """
    将日期格式转换为时间戳
    PS: 使用numpy保存时间戳，如果要精确到毫秒，则使用float64；如果仅需要精确到秒，则使用int64
    float32和int32会导致精度丢失
    """
    try:
        stamp = time.mktime(time.strptime(date, format))
    except:
        stamp = -1
    return stamp


def stamp2date(stamp, format='%Y-%m-%d %H:%M:%S') -> str:
    """将时间戳转换为日期格式"""
    try:
        date = datetime.fromtimestamp(stamp).strftime(format)
    except:
        date = ""
    return date


class Recorder(object):
    def __init__(self):
        self.content: Dict = {}

    def push(self, key, v, start=1):
        if not key in self.content:
            self.content[key] = {"avg": 0, "total": 0, "current": 0, "nums": 0, "count": 0}
        info: dict = self.content[key]
        info["count"] = info["count"] + 1
        if info["count"] < start:
            info.update({"avg": 0, "total": 0, "current": 0, "nums": 0})
        else:
            info["nums"] = info["nums"] + 1
            info["total"] = info["total"] + v
            info["current"] = v
            info["avg"] = info["total"] / info["nums"]

    def get(self, key) -> Dict:
        return self.content[key]

    def get_count(self, key) -> int:
        if key not in self.content: return 0
        return self.content[key].get("count", 0)

    def reset(self):
        self.content: Dict = {}

    def info(self, key):
        print(self.get(key))


recorder = Recorder()


def performance(tag="", n=1, log=print):
    """
    :param tag:
    :param n: 从第几次开始记录数据
    :return:
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # torch.cuda.synchronize()
            t0 = time.time()
            result = func(*args, **kwargs)
            # torch.cuda.synchronize()
            t1 = time.time()
            key = tag if tag else str(func.__name__)
            recorder.push(key=key, v=(t1 - t0) * 1000, start=n)
            content = recorder.get(key)
            info = ["{}:{:.3f}ms".format(n, content.get(n, 0)) for n in ["current", "avg", "total"]]
            info += ["count:{}".format(content['count'])]
            elapsed = "\t ".join(info)
            if tag:
                log("{:20s}{:20s} elapsed: {}".format(tag, func.__name__, elapsed))
            else:
                log("{:20s} elapsed: {}".format(func.__name__, elapsed))
            return result

        return wrapper

    return decorator


class Performance(object):
    def __init__(self, tag="", n=1, log=print):
        """
        :param tag:
        :param n: 从第几次开始记录数据
        :return:
        """
        self.tag = tag
        self.n = n
        self.log = log

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # 参数为异常信息
        self.t1 = time.time()
        recorder.push(key=self.tag, v=(self.t1 - self.t0) * 1000, start=self.n)
        self.info(key=self.tag)

    def info(self, key):
        content = recorder.get(key)
        info = ["{}:{:.3f}ms".format(n, content.get(n, 0)) for n in ["current", "avg", "total"]]
        info += ["count:{}".format(content['count'])]
        elapsed = "\t ".join(info)
        tag_ = f"{self.tag} " if self.tag else ""
        self.log("{:20s}elapsed: {}\t".format(tag_, elapsed))

    def task(self):
        pass


@performance("test1", n=5)
def targe_func1():
    time.sleep(1)


@performance("test111111")
def targe_func2():
    time.sleep(0.5)


def targe_func3():
    with Performance("test222", n=5) as p:
        time.sleep(1)


def targe_func4():
    with Performance("test22222222") as p:
        time.sleep(0.5)


def targe_func():
    targe_func1()
    # targe_func2()
    # targe_func3()
    # targe_func4()


if __name__ == '__main__':
    for i in range(10):
        targe_func()
