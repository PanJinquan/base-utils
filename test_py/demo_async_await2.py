# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-25 11:47:17
    @Brief  : <Python — 异步async/await> https://blog.csdn.net/weixin_45804031/article/details/124579021
"""
import os
import asyncio
import requests
from typing import Callable
import time
import asyncio
import requests
from concurrent import futures
from pybaseutils import time_utils
executor = futures.ThreadPoolExecutor(max_workers=5)


def task_process(params):
    time.sleep(1)
    print(params)
    params["data"] = params["data"] + "/1"
    time.sleep(1)
    return params


async def async_task(params, task: Callable):
    def run_in_executor(params):
        params = task(params)
        return params

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, run_in_executor, params)


async def http_api(params, task: Callable):
    result = await async_task(params, task=task)
    return result


@time_utils.performance("test_time")
def test_time():
    url = 'https://www.baidu.com'
    for i in range(0, 3):
        params = {"data": url, "n": i}
        r = task_process(params)


@time_utils.performance("async_test_time")
def async_test_time():
    tasks = []
    url = 'https://www.baidu.com'
    for i in range(0, 3):
        params = {"data": url, "id": i}
        t = http_api(params, task=task_process)
        tasks.append(t)
    loop = asyncio.get_event_loop()
    rs, s = loop.run_until_complete(asyncio.wait(tasks))
    result = [r.result() for r in rs]
    loop.close()
    print(result)


if __name__ == "__main__":
    async_test_time()
    # test_time()
