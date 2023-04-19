# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-04-25 16:20:17
    @Brief  :
"""
import threading
import time
import random
from pybaseutils.worker import Worker, Compose
from pybaseutils.thread_utils import thread_lock_decorator, thread_lock
from pybaseutils import debug
from typing import List, Callable


class Producer(Worker):
    def __init__(self, num_worker=2, maxsize=5, *args, **kwargs):
        super(Producer, self).__init__(num_worker=num_worker, maxsize=maxsize, args=args, kwargs=kwargs)
        self.nums = kwargs["nums"]

    def task(self, ):
        """发送数据"""
        done = True
        count = 0
        while done:
            # time.sleep(1)  # 每隔一段时间发送一次数据
            data = "{}.jpg".format(count)
            data = {"index": count, "data": data}
            self.output.put(data)
            print("{},发送数据：{}".format(self.info(), data))
            done = self.nums > count
            count += 1

    def start(self, *args, **kwargs):
        t = threading.Thread(target=self.task, args=args, kwargs=kwargs)
        # 执行线程
        t.start()


class Consumer1(Worker):
    def __init__(self, num_worker=2, maxsize=5, *args, **kwargs):
        super(Consumer1, self).__init__(num_worker=num_worker, maxsize=maxsize, args=args, kwargs=kwargs)

    # @thread_lock_decorator()
    def task(self, data):
        # data.update({"C2": 2})
        t = random.uniform(0, 2)
        time.sleep(t)
        print("{},处理数据：{}".format(self.info(), data))
        return data


class Consumer2(Worker):
    def __init__(self, num_worker=2, maxsize=5, *args, **kwargs):
        super(Consumer2, self).__init__(num_worker=num_worker, maxsize=maxsize, args=args, kwargs=kwargs)

    def task(self, data):
        t = random.uniform(0, 3)
        time.sleep(t)
        print("{},处理数据：{}".format(self.info(), data))
        return data


class Result(Worker):
    def __init__(self, num_worker=2, maxsize=5, *args, **kwargs):
        super(Result, self).__init__(num_worker=num_worker, maxsize=maxsize, args=args, kwargs=kwargs)
        self.nums = kwargs["nums"]

    def task(self):
        data = self.input.get(block=True, timeout=None)
        return data

    @debug.run_time_decorator()
    def start(self, ):
        """线程任务"""
        result = []
        while self.nums >= len(result):
            try:
                data = self.task()
                result.append(data)
                print("{}".format(self.info()))
            except Exception as e:
                pass
            finally:
                pass
        print("result:{}".format(result))
        return result


if __name__ == "__main__":
    """
    _target_v1 15075.741529464722ms 15066.256046295166ms 
    _target_v2 13534.188032150269ms 13520.671367645264ms 
    """
    nums = 10
    consumer = Compose([
        Producer(num_worker=4, maxsize=5, nums=nums),
        Consumer1(num_worker=8, maxsize=5),
        Consumer2(num_worker=4, maxsize=5),
        Result(num_worker=4, maxsize=5, nums=nums),
    ])
    consumer.start()
