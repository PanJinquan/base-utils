# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-04-25 16:20:17
    @Brief  :
"""
import threading
import time
import random
from abc import ABC

from pybaseutils.worker import Worker, as_completed
from pybaseutils import debug


class Producer(Worker):
    def __init__(self, input=None, num_worker=2, maxsize=5):
        super(Producer, self).__init__(input, num_worker=num_worker, maxsize=maxsize)

    def task(self, nums):
        """发送数据"""
        done = True
        count = 0
        while done:
            time.sleep(0.01)  # 每隔一段时间发送一次数据
            data = "{}.jpg".format(count)
            data = {"index": count, "data": data}
            self.output.put(data)
            print("{},发送数据：{}".format(self.info(), data))
            done = nums > count
            count += 1

    def start(self, nums=10):
        t = threading.Thread(target=self.task, args=(nums,))
        # 执行线程
        t.start()


class Consumer1(Worker):
    def __init__(self, input=None, num_worker=2, maxsize=5):
        """
        :param input:  输入队列
        :param num_worker:  线程池最大线程数
        :param maxsize:
        """
        super(Consumer1, self).__init__(input, num_worker=num_worker, maxsize=maxsize)

    def task(self, data):
        # data.update({"C2": 2})
        t = random.uniform(0, 4)
        time.sleep(t)
        print("{},处理数据：{}".format(self.info(), data))
        return data


class Consumer2(Worker):
    def __init__(self, input=None, num_worker=2, maxsize=5):
        """
        :param input:  输入队列
        :param num_worker:  线程池最大线程数
        :param maxsize:
        """
        super(Consumer2, self).__init__(input, num_worker=num_worker, maxsize=maxsize)

    def task(self, data):
        t = random.uniform(0, 4)
        time.sleep(t)
        print("{},处理数据：{}".format(self.info(), data))
        return data


class Result(Worker):
    def __init__(self, input=None, num_worker=2, maxsize=5):
        """
        :param input:  输入队列
        :param num_worker:  线程池最大线程数
        :param maxsize:
        """
        super(Result, self).__init__(input, num_worker=num_worker, maxsize=maxsize)

    def task(self):
        data = self.input.get(block=True, timeout=None)
        return data

    @debug.run_time_decorator()
    def start(self, nums):
        """线程任务"""
        result = []
        while nums > len(result):
            try:
                data = self.task()
                result.append(data)
                print("{}".format(self.info()))
            except Exception as e:
                pass
            finally:
                pass
        return result


if __name__ == "__main__":
    """
    _target_v1 15075.741529464722ms 15066.256046295166ms 
    _target_v2 13534.188032150269ms 13520.671367645264ms 
    """
    nums = 10
    p = Producer(input=None, num_worker=4, maxsize=5)
    c1 = Consumer1(input=p, num_worker=4, maxsize=5)
    c2 = Consumer2(input=c1, num_worker=4, maxsize=5)
    r = Result(input=c2, num_worker=4, maxsize=5)
    p.start(nums=nums)
    c1.start()
    c2.start()
    result = r.start(nums=nums)
    print(result)
