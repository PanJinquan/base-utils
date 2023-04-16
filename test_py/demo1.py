# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import asyncio
import time
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils


async def say_after(delay, what):
    await asyncio.sleep(delay)
    # time.sleep(delay)
    print(what)


async def main():
    print(f"started at {time.strftime('%X')}")

    # say_after(2, 'hello')
    # await say_after(1, 'hello')执行完之后，才继续向下执行
    # say_after(1, 'world')

    print(f"finished at {time.strftime('%X')}")


if __name__ == "__main__":
    # asyncio.run(main()) # python3.7
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
