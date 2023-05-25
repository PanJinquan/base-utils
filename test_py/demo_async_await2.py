# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-25 11:47:17
    @Brief  : <Python — 异步async/await> https://blog.csdn.net/weixin_45804031/article/details/124579021
"""
import asyncio
import requests
import time
import asyncio
import requests
from concurrent import futures

executor = futures.ThreadPoolExecutor(max_workers=5)


def task(params):
    time.sleep(1)
    print(params)
    time.sleep(1)
    return params


async def async_upload_image(params):
    # 将content字段中所有的文件和图片资源都上传到服务器，返回url代替
    def run_in_executor(params):
        params = task(params)
        return params

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, run_in_executor, params)


async def http_api(params):
    result = await async_upload_image(params)
    return result


if __name__ == "__main__":
    tasks = []
    url = 'https://www.baidu.com'
    for i in range(0, 3):
        params = {"data": url, "n": i}
        r = http_api(params)
        tasks.append(r)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    print(tasks)
