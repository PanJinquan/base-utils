# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-10-29 16:52:22
# @Brief  :
# --------------------------------------------------------
"""
import os
import numpy as np
import time
import threading
from typing import List, Dict, Callable
from pybaseutils import thread_utils, file_utils, image_utils, base64_utils
from pybaseutils.cvutils import video_utils
from pybaseutils.base import list_queue

thread_lock = threading.Lock()


class ProducerConsumer:
    def __init__(self, winsize=3, overlap=0.0, max_workers=2, **kwargs) -> None:
        """
        :param winsize:
        :param overlap:
        :param max_workers:
        :param kwargs:
        """
        self.tag = self.__class__.__name__
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.winsize = winsize
        self.overlap = overlap
        self.max_workers = max_workers
        self.kwargs = kwargs
        self.producer = list_queue.Queue(maxsize=self.winsize)  # 生产者
        self.consumer = list_queue.Queue(maxsize=self.winsize)  # 消费者
        self.producer_end = False  # 生产者是否结束
        self.consumer_end = False  # 消费者是否结束

    def stop_producer(self):
        """
        停止生产者线程
        :return:
        """
        self.producer_end = True
        self.log("stop producer")

    def stop_consumer(self):
        """
        停止消费者线程
        :return:
        """
        self.consumer_end = True
        self.log("stop consumer")

    def task(self, data: List, *args, **kwargs):
        """
        任务函数，处理输入数据
        :param data: 输入数据
        :return: 处理结果
        """
        self.log(f"args  ={args},kwargs={kwargs}")
        time.sleep(3)
        result = "收到消息：" + file_utils.get_time(format="s")
        return result

    def task_consumer(self, task, *args, **kwargs):
        """消费者任务"""
        self.log(f"start consumer thread")
        self.consumer_end = False
        pool = thread_utils.ThreadPool(max_workers=self.max_workers)
        future = []
        while True:
            # winsize = self.producer.qsize() if self.producer_end else self.winsize
            if self.producer_end and self.producer.qsize() <= self.winsize:
                winsize, overlap = self.producer.qsize(), 0  # TODO 如果生产者已经结束，则一次性处理剩余的数据
            else:
                winsize, overlap = self.winsize, self.overlap
            while self.producer.qsize() >= winsize > 0 and len(future) < self.max_workers:
                data = self.producer.get_window(winsize=winsize, overlap=overlap, block=False)  # 提取窗口数据
                f = pool.submit(task, data, args, **kwargs)
                future.append(f)
                self.log(f"tid={id(f)} producer={self.producer.qsize()} consumer={self.consumer.qsize()} "
                         f"winsize={winsize} future={len(future)} ")
            if not future: time.sleep(0.1)
            tmp = []
            for f in future:
                try:
                    self.consumer.put(f.result(timeout=0.1))
                except Exception as e:
                    tmp.append(f)
            future = tmp
            # 生产者队列空，且停止发送数据(pend=True)
            if len(future) == 0 and self.producer_end and self.producer.qsize() == 0:
                print(f"finish consumer thread")
                break
        self.stop_consumer()  # 消费者停止处理数据

    def start_consumer(self, task: Callable, *args, **kwargs):
        """启动消费者线程
        :param task: 任务函数
        :param args: 任务函数参数
        :param kwargs: 任务函数参数
        :return:
        """
        thread = threading.Thread(target=self.task_consumer, args=args, kwargs={"task": task, **kwargs})
        thread.daemon = True
        thread.start()


def example(text, video, realtime=False, freq=2, wintime=3, overlap=0.5, max_workers=2, delay=20, vis=True, **kwargs):
    """
    实时分析功能，从视频中提取帧进行分析
    :param text: 用户问题/任务描述
    :param video: 视频文件路径或者摄像头ID
    :param realtime: 是否实时处理视频帧，实时处理会丢弃很多视频帧，非实时会处理所有视频帧，但视频播放会很卡顿
    :param freq: 视频帧提取频率（Hz）
    :param wintime:  窗口时间长度，单位秒
    :param overlap: 滑动窗口重叠比例
    :param max_workers: 视频处理线程数
    :return:
    """
    winsize = int(wintime * freq)
    pc = ProducerConsumer(winsize=winsize, overlap=overlap, max_workers=max_workers, **kwargs)
    # TODO 打开视频文件或摄像头
    w, h, num_frames, fps = image_utils.get_video_info(video)
    interval = int(fps / freq) if fps > 0 else 1
    video_cap = video_utils.video_iterator(video, save_video=None)
    pc.start_consumer(task=pc.task, text=text, freq=freq)  # 启动消费者线程
    # TODO 主线程处理视频帧
    for data_info in video_cap:
        image = data_info["frame"][:, :, ::-1]  # BGR to RGB
        image = image_utils.resize_image(image, size=(None, 640))
        data_info["frame"] = image
        # TODO 视频抽帧，放入生产者队列，非实时处理会阻塞，实时处理会丢弃旧的视频帧
        if data_info['count'] % interval == 0:
            # data_info['text'] = text
            # data_info['freq'] = freq
            pc.producer.put(data_info, block=not realtime)
        if vis: image_utils.show_image("image", image, delay=delay, use_rgb=True)
        if data_info["finish"]: pc.stop_producer()  # 标记生产者是否结束
        while pc.consumer.qsize() > 0 or (pc.producer_end and not pc.consumer_end):
            result = pc.consumer.pop(block=False, timeout=0.005)
            if result: print("result={}".format(result))


if __name__ == '__main__':
    text = "请分析这个视频"
    video_file = "/home/PKing/Videos/video4.mp4"
    example(text=text, video=video_file, freq=2)
