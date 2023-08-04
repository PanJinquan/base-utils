# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail : 
    @Date   : 2023-08-01 10:27:56
    @Brief  :
"""
import numpy as np
import librosa


def read_audio(audio_file, sr=16000, mono=True):
    """
    默认将多声道音频文件转换为单声道，并返回一维数组；
    如果你需要处理多声道音频文件，可以使用 mono=False,参数来保留所有声道，并返回二维数组。
    :param audio_file:
    :param sr: sampling rate
    :param mono: 设置为true是单通道，否则是双通道
    :return:
    """
    audio_data, sr = librosa.load(audio_file, sr=sr, mono=mono)
    audio_data = audio_data.T.reshape(-1)
    return audio_data, sr


def print_vector(name, data):
    np.set_printoptions(precision=7, suppress=False)
    print("------------------------%s------------------------\n" % name)
    print("{}".format(data.tolist()))


if __name__ == '__main__':
    sr = None
    audio_file = "/media/PKing/新加卷1/SDK/audio/Librosa-Cpp/data/data_s1.wav"
    data, sr = read_audio(audio_file, sr=sr, mono=False)
    print("sr         = %d, data size=%d" % (sr, len(data)))
    print_vector("audio data", data)
