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
from pybaseutils import file_utils, time_utils
from pybaseutils.cvutils import video_utils, camera
from pybaseutils.base import list_queue


class ProducerConsumer(object):
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
        self.que: list_queue.Queue = None
        self.con: threading.Thread = None
        self.pro: threading.Thread = None
        self.state: int = 0  # TODO
        self.state_info = {-1: '正在终止producer&consumer', 0: 'producer&consumer已终止', 1: 'running'}

    def start(self, **kwargs):
        if self.running(): return  # TODO 系统正在运行中，不能重复启动
        self.state = 1
        self.cfg = {**self.cfg, **kwargs}
        self.que = list_queue.Queue(maxsize=self.cfg['wsize'] * 2)  # 生产者
        self.pro = threading.Thread(target=self.producer, kwargs=self.cfg)
        self.pro.daemon = True  # 跟随主进程一起退出
        self.pro.start()
        self.con = threading.Thread(target=self.consumer, kwargs=self.cfg)
        self.con.daemon = True  # 跟随主进程一起退出
        self.con.start()
        self.log("{:10s} start success  ={}".format(self.tag, self.cfg))

    def running(self):
        """获得系统运行状态"""
        return self.state == 1

    def stop(self, **kwargs):
        """终止生产者和消费者"""
        self.log("{:10s} stop   operation".format(self.tag))
        try:
            self.state = -1
            if self.pro: self.pro.join(timeout=5)
        except Exception as e:
            traceback.print_exc()
        self.pro = None
        self.log("{:10s} stop   producer   ={}".format(self.tag, self.pro))

        try:
            self.state = -1
            if self.con: self.con.join(timeout=5)
        except Exception as e:
            traceback.print_exc()
        self.con = None
        self.log("{:10s} stop   consumer   ={}".format(self.tag, self.con))
        self.state = 0
        self.que.clear()

    def consumer_tasks(self, data: List, **kwargs):
        """
        消费者任务函数
        :param data: 生成者发送的数据，List,数量由wsize确定,data=[info,info,...]
        :return: 处理结果
        """
        result = [self.consumer_task(info, **kwargs) for info in data]
        return result

    def consumer_task(self, info: Dict, **kwargs):
        """
        消费者任务函数
        p0生产者读取视频前的时间戳，ps生产者读取视频后的时间戳,pn生产者发送给消费者后的时间
        cs消费者接收到数据的时间戳，cn消费者处理完数据的时间戳
        :param info: 生成者发送的数据，
        :return: 处理结果
        """
        time_utils.cpu_elapse(1.0)  # 模拟CPU计算耗时
        result = "收到消息：" + file_utils.get_time(format="s")
        self.log(f"{self.tag:10s} result={result},data nums={info.keys()},time={info['time']}")
        return result

    def consumer(self, **kwargs):
        """
        消费者
        :param kwargs:
        :return:
        """
        self.log("{:10s} run consumer   ={}".format(self.tag, self.con))
        while True:
            try:
                if self.state <= 0: break
                # info = self.queue.pop() # 返回一个元素
                data = self.que.get_window(wsize=self.cfg['wsize'], overlap=self.cfg['overlap'])  # 返回一个元素
                if not data: continue
                cs = time.time()  # 消费者接收到数据的时间戳
                ct = self.consumer_tasks(data)
                cn = time.time()  # 消费者处理完数据的时间戳
                count = [info['count'] for info in data]
                for c in count:
                    if c % self.cfg['log_freq'] == 0:
                        co = [f"{co:06d}" for co in count]
                        self.log(f'{self.tag:10s} consumer id={co} csn={self.elap(cn, cs)} q={self.que.qsize()}')
                        break
            except Exception as e:
                traceback.print_exc()
        self.que.clear()
        self.state = 0
        self.log("{:10s} finish consumer   ={}".format(self.tag, self.con))

    def producer_task(self, info: Dict, **kwargs):
        """
        生产者任务函数
        :param info: 生成者发送的数据，List,数量由wsize确定,{'frame': frame, 'size': size, "count": count, 'duration': cap.duration, 'time': t1}
        :return: 处理结果
        """
        return info

    def producer(self, video: str | int, block=True, **kwargs):
        """
        生产者
        :param video:
        :param block:  是否阻塞，阻塞不会丢弃队列数据，非阻塞即实时处理，会丢弃队列数据
        :param kwargs:
        :return:
        """
        self.log("{:10s} block          ={}".format(self.tag, block))
        self.log("{:10s} config         ={}".format(self.tag, self.cfg))
        self.log("{:10s} run producer   ={}".format(self.tag, self.pro))
        if file_utils.is_number(str(video)): video = video_utils.get_usb_camera(int(video))  # 获取当前系统中连接的USB摄像头索引
        cap = camera.CameraCapture(video=video, size=self.cfg['size'], scale=self.cfg['scale'], fps=self.cfg['fps'],
                                   input_format=self.cfg['input_format'])
        # TODO 主线程处理视频帧
        count = 0
        while True:
            try:
                # TODO p0生产者读取视频前的时间戳，ps生产者读取视频后的时间戳,pn生产者发送给消费者后的时间
                p0 = time.time()
                ret, frame = cap.read()
                ps = time.time()
                if self.state <= 0 or not ret:
                    self.state = -1
                    self.que.put({}, block=block)  # TODO 发送空数据，以确保consumer结束
                    cap.stop()
                    break
                size = [frame.shape[1], frame.shape[0]]  # 视频帧的宽度和高度
                info = {'frame': frame, 'size': size, "count": count, 'duration': cap.duration,
                        'time': dict(p0=p0, ps=ps)}
                info = self.producer_task(info)
                # TODO 发送给消费者
                self.que.put(info, block=block)  # TODO 当block=True时，若消费者卡顿，会导致这里阻塞，且无法终止
                pn = time.time()
                if count % self.cfg['log_freq'] == 0:
                    self.log(
                        f'{self.tag:10s} producer id={[f"{count:06d}"]} p0s={self.elap(ps, p0)} p0n={self.elap(pn, p0)}')
                count += 1
            except Exception as e:
                traceback.print_exc()
        self.que.clear()
        self.state = 0
        self.log("{:10s} finish producer   ={}".format(self.tag, self.pro))
        return 0

    @staticmethod
    def elap(t1, t0):
        """计算耗时"""
        return f"{(t1 - t0) * 1000:.0f}ms"


if __name__ == '__main__':
    cfg = {
        # "video": "/media/PKing/dev2/SDK/base-utils/data/video/kunkun_cut.mp4",
        "video": "/home/PKing/Downloads/一号.mp4",
        "fps": 2,
        "wsize": 10,  #
        "overlap": 0.5,
        # "size": (1920, 1080),
        # "size": (1280, 720),
        "size": (),
        "scale": 1.0,
        "log_freq": 1,  # 日志打印频率,
        # "input_format": "mjpeg",  # 摄像头输入格式 "mjpeg"、"yuyv422"
        "input_format": "yuyv422",  # 摄像头输入格式 "mjpeg"、"yuyv422"
    }
    pc = ProducerConsumer(log=print, **cfg)
    pc.start(**cfg)
    time.sleep(50)
    pc.stop()
    time.sleep(2)
    print("done")
