# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils, plot_utils, numpy_utils
from pybaseutils.audio import audio_utils

import pyaudio
import wave
import time

file = "/home/PKing/Videos/dde-introduction.wav"
# 打开WAV文档
wf = wave.open(file, "rb")

# 实例化一个PyAudio对象
pa = pyaudio.PyAudio()
# 打开声卡
stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),  # 从wf中获取采样深度
                 channels=wf.getnchannels(),  # 从wf中获取声道数
                 rate=wf.getframerate(),  # 从wf中获取采样率
                 output=True)  # 设置为输出
count = 0
while count < 8 * 500:
    time.sleep(0.001)
    data = wf.readframes(15)
    stream.write(data)
