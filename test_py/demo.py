# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import time
from pybaseutils import time_utils


@time_utils.performance("demo_map")
def demo_map(inp):
    data = list(map(int, inp))
    # data = list(map(lambda x: int(x), inp))
    return data


@time_utils.performance("demo_for1")
def demo_for1(inp):
    data = []
    for d in inp:
        data.append(int(d))
    return data


@time_utils.performance("demo_for2")
def demo_for2(inp):
    data = [int(d) for d in inp]
    return data


if __name__ == "__main__":
    inp = "123456789" * 10000
    data0 = demo_map(inp)
    data1 = demo_for1(inp)
    data2 = demo_for2(inp)
    print(len(data1))
    print(len(data2))
