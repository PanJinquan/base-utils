# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-03-03 17:48:38
# @Brief  :
# --------------------------------------------------------
"""
import av
import cv2
import pyaudio
import numpy as np
import threading
import traceback
from queue import Queue


class MediaLibrary(object):
    def __init__(self, url):
        """
        :param url:
        """
        self.url = url
        # 初始化PyAudio
        self.container = av.open(url, timeout=10)  # 打开RTSP流
        self.audio = pyaudio.PyAudio()
        # 启动音频线程
        self.audio_queue = Queue(maxsize=50)  # 音频帧缓冲区
        self.audio_thread = threading.Thread(target=self.audio_task)
        self.audio_thread.start()
        self.media_info = dict()

    def play_image(self, image, title="video", delay=5):
        """
        播放图片
        :param image:
        :param title:
        :param delay:
        :return:
        """
        # cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
        # cv2.imshow(title, image)
        cv2.waitKey(delay)

    def play_audio(self, frame):
        """
        播放音频
        :param frame:
        :return:
        """
        self.audio_queue.put(frame)

    def audio_task(self):
        """
        音频播放线程
        :return:
        """
        stream = None
        while True:
            frame = self.audio_queue.get()
            if frame is None: break
            if stream is None:
                c = len(frame.layout.channels)
                r = frame.rate
                stream = self.audio.open(format=pyaudio.paFloat32, channels=c, rate=r, output=True)
            # 音频处理
            data = frame.to_ndarray().astype(np.float32).tobytes()
            stream.write(data)
        stream.stop_stream()
        stream.close()

    def display(self, title="video", delay=5):
        """
        播放视频
        :param title:
        :param delay:
        :return:
        """
        try:
            for i, frame in enumerate(self.container.decode(video=0, audio=0)):  # 遍历容器中的包（音视频分离）
                if isinstance(frame, av.VideoFrame):
                    self.media_info["width"] = frame.width
                    self.media_info["height"] = frame.height
                    self.play_image(image=frame.to_ndarray(format='bgr24'), title=title, delay=delay)
                elif isinstance(frame, av.AudioFrame):
                    self.media_info["channels"] = len(frame.layout.channels)
                    self.media_info["rate"] = frame.rate
                    self.play_audio(frame=frame)
                if i == 3: print(self.media_info)
        except Exception as e:
            traceback.print_exc()
        finally:
            # 清理资源
            cv2.destroyAllWindows()
            self.audio_queue.put(None)
            self.audio_thread.join()
            self.container.close()
            self.audio.terminate()


if __name__ == "__main__":
    # RTSP流地址
    # url = "rtsp://admin:C2332416@192.168.2.35"
    url = "/home/PKing/Videos/dde-introduction.mp4"
    # url = "E:/project/dde-introduction.mp4"
    m = MediaLibrary(url)
    m.display()
