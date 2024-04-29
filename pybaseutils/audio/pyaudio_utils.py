# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import time
import os
import pyaudio
import wave
from tqdm import tqdm


def play_audio(wave_file):
    """
    播放音频
    :param wave_file:
    :return:
    """
    CHUNK = 1024
    wf = wave.open(wave_file, 'rb')
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data
    data = wf.readframes(CHUNK)
    # play stream (3)
    datas = []
    while len(data) > 0:
        data = wf.readframes(CHUNK)
        datas.append(data)

    for d in tqdm(datas):
        stream.write(d)
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()


def record_audio(wave_file, time):
    """
    录音
    :param wave_file:
    :param time:
    :return:
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    wf = wave.open(wave_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("* recording")
    for i in tqdm(range(0, int(RATE / CHUNK * time))):
        data = stream.read(CHUNK)
        wf.writeframes(data)

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


if __name__ == "__main__":
    file = "output.wav"
    record_audio(file, time=4)
    play_audio(file)
