# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import threading
import time
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

class ProcessPool(object):
    def __init__(self):
        self.pool = Pool(4)

    def worker(self, x):
        print(x) # Should be printing "testing123"

    def run(self):
        res = self.pool.apply_async(self.worker, ("testing123",))
        print(res.get()) # NotImplementedError
        self.pool.close()
        self.pool.join()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


sandbox = ProcessPool()
sandbox.run()


sandbox = ProcessPool()
sandbox.run()