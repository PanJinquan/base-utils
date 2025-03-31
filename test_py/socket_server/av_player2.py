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
        self.video_stream = next(s for s in self.container.streams if s.type == 'video')
        self.audio_stream = next(s for s in self.container.streams if s.type == 'audio')
        self.media_info = dict(width=self.video_stream.width,
                               height=self.video_stream.height,
                               channels=self.audio_stream.channels,
                               rate=self.audio_stream.rate
                               )
        self.audio = pyaudio.PyAudio()
        # # 启动音频线程
        self.audio_queue = Queue(maxsize=50)  # 音频帧缓冲区
        self.video_frame = None
        self.audio_frame = None
        self.audio_thread = threading.Thread(target=self.audio_loop)
        self.video_thread = threading.Thread(target=self.video_loop)
        self.audio_thread.start()
        self.video_thread.start()
        print(self.media_info)

    def audio_loop(self):
        """
        音频播放线程
        :return:
        """
        stream = self.audio.open(format=pyaudio.paFloat32, channels=self.media_info['channels'],
                                 rate=self.media_info['rate'], output=True)
        while True:
            frame = self.audio_queue.get()
            if frame is None: break
            data = frame.to_ndarray().astype(np.float32).tobytes()
            stream.write(data)
        stream.stop_stream()
        stream.close()

    def video_loop(self):
        """
        音频播放线程
        :return:
        """
        delay = 15
        title = "video"
        while True:
            if self.video_frame is None: continue
            # cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
            # cv2.imshow(title, self.video_frame)
            cv2.waitKey(delay)

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
                    self.video_frame = frame.to_ndarray(format='bgr24')
                elif isinstance(frame, av.AudioFrame):
                    self.audio_queue.put(frame)
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
