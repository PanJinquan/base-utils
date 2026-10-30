"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-10-29 12:21:07
# @Brief  : 摄像头/视频生产者(Producer)和消费者(Consumer)
# --------------------------------------------------------
"""
import os
import numpy as np
import time
import threading
import traceback
from typing import List, Dict, Callable
from pybaseutils import file_utils
from pybaseutils.cvutils import video_utils, camera
from pybaseutils.base import list_queue


class CameraPC(object):
    def __init__(self, **kwargs):
        """
        :param kwargs:  配置参数
        {
            "size": (1920, 1080),        # 图像分辨率(w,h)， size=()表示原始分辨率
            "video": "video.mp4",        # 视频ID或者视频文件
            "fps": 10,                   # 抽帧频率
            "wsize": 10,                 # 获得数据的窗口大小(相当于batch_size)
            "scale": 1.0,                # 图像缩放倍率
            "overlap": 0.5,              # 获得数据重叠率
            "log_freq": 1,               # 日志打印频率,
            "input_format": "yuyv422",   # 摄像头输入格式 "mjpeg"、"yuyv422"
        }
        """
        self.tag = self.__class__.__name__
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.cfg = kwargs
        self.state: int = 1  # TODO
        self.queue: list_queue.Queue = None
        self.consumer: threading.Thread = None
        self.producer: threading.Thread = None
        self.state_info = {-1: '正在终止producer&consumer', 0: 'producer&consumer已终止', 1: 'running'}

    def start(self, **kwargs):
        self.cfg = {**self.cfg, **kwargs}
        self.queue = list_queue.Queue(maxsize=self.cfg['wsize'] * 2)  # 生产者
        self.producer = threading.Thread(target=self.target_producer, kwargs=self.cfg)
        self.producer.daemon = True  # 跟随主进程一起退出
        self.producer.start()
        self.consumer = threading.Thread(target=self.target_consumer, kwargs=self.cfg)
        self.consumer.daemon = True  # 跟随主进程一起退出
        self.consumer.start()

    def stop(self, **kwargs):
        """终止生产者和消费者"""
        self.log("{:10s}     finish operation".format(self.tag))
        try:
            self.state = -1
            if self.producer: self.producer.join(timeout=5)
        except Exception as e:
            traceback.print_exc()
        self.producer = None
        self.log("{:10s}     finish producer   ={}".format(self.tag, self.producer))

        try:
            self.state = -1
            if self.consumer: self.consumer.join(timeout=5)
        except Exception as e:
            traceback.print_exc()
        self.consumer = None
        self.log("{:10s}     finish consumer   ={}".format(self.tag, self.consumer))
        self.state = 0
        self.queue.clear()

    def task(self, data: List, *args, **kwargs):
        """
        消费者任务函数，处理输入数据
        :param data: 生成者发送的数据，List,数量由winsize确定
        :return: 处理结果
        """
        time.sleep(0.1)
        result = "收到消息：" + file_utils.get_time(format="s")
        self.log(f"result={result},data nums={len(data)}")
        return result

    def target_consumer(self, **kwargs):
        """
        消费者
        :param kwargs:
        :return:
        """
        self.log("{:10s}      run consumer   ={}".format(self.tag, self.consumer))
        while True:
            try:
                if self.state <= 0: break
                # info = self.queue.pop() # 返回一个元素
                data = self.queue.get_window(wsize=self.cfg['wsize'], overlap=self.cfg['overlap'])  # 返回一个元素
                if not data: continue
                t2 = time.time()  # 消费者接收到数据的时间
                rt = self.task(data, **self.cfg)
                t3 = time.time()  # 模型推理完的时间戳
                count = [f"{info['count']:06d}" for info in data]
                self.log(f'consumer id={count} t23={self.elap(t3, t2)} q={self.queue.qsize()}')
            except Exception as e:
                traceback.print_exc()
        self.queue.clear()
        self.state = 0
        self.log("{:10s}       stop consumer   ={}".format(self.tag, self.consumer))

    def target_producer(self, video: str | int, block=False, **kwargs):
        """
        生产者
        :param video:
        :param block:  是否阻塞，阻塞不会丢弃队列数据，非阻塞即实时处理，会丢弃队列数据
        :param kwargs:
        :return:
        """
        self.log("{:10s}      config         ={}".format(self.tag, self.cfg))
        self.log("{:10s}      run producer   ={}".format(self.tag, self.producer))
        if isinstance(video, int): video = video_utils.get_usb_camera(video)  # 获取当前系统中连接的USB摄像头索引
        cap = camera.CameraCapture(video=video, size=self.cfg['size'], scale=self.cfg['scale'], fps=self.cfg['fps'],
                                   input_format=self.cfg['input_format'])
        # TODO 主线程处理视频帧
        count = 0
        while True:
            try:
                # TODO t0 是读取视频前的时间戳，t1 是读取视频后的时间戳
                t0 = time.time()
                ret, frame = cap.read()
                t1 = time.time()
                if self.state <= 0 or not ret:
                    self.state = -1
                    self.queue.put({}, block=block)  # TODO 发送空数据，以确保consumer结束
                    cap.stop()
                    break
                size = [frame.shape[1], frame.shape[0]]  # 视频帧的宽度和高度
                info = {'frame': frame, 'size': size, "count": count}
                # TODO 发送给消费者
                self.queue.put(info, block=block)  # TODO 当realtime=False时，若消费者卡顿，会导致这里阻塞，且无法终止
                t2 = time.time()
                if count % self.cfg['log_freq'] == 0:
                    self.log(f'producer id={[f"{count:06d}"]} t10={self.elap(t1, t0)} t20={self.elap(t2, t0)}')
                count += 1
            except Exception as e:
                traceback.print_exc()
        self.queue.clear()
        self.state = 0
        self.log("{:10s}       stop producer   ={}".format(self.tag, self.producer))
        return 0

    def elap(self, t1, t0):
        """计算耗时"""
        return f"{(t1 - t0) * 1000:.0f}ms"


if __name__ == '__main__':
    cfg = {
        "video": "/media/PKing/dev2/SDK/base-utils/data/video/kunkun_cut.mp4",
        "fps": 10,
        "wsize": 5,  #
        "overlap": 0.5,
        # "size": (1920, 1080),
        # "size": (1280, 720),
        "size": (),
        "scale": 1.0,
        "log_freq": 1,  # 日志打印频率,
        # "input_format": "mjpeg",  # 摄像头输入格式 "mjpeg"、"yuyv422"
        "input_format": "yuyv422",  # 摄像头输入格式 "mjpeg"、"yuyv422"
    }
    cpc = CameraPC(log=print, **cfg)
    cpc.start(**cfg)
    time.sleep(50)
    cpc.stop()
    time.sleep(2)
    print("done")
