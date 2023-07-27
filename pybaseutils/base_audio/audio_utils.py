# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-11 10:36:28
    @Brief  : pip install ffmpy -i https://pypi.douban.com/simple
              https://blog.csdn.net/zhiweihongyan1/article/details/121735158
              安装依赖项:见install.sh
"""
import os
import base64
import io
import soundfile
import numpy as np
import copy
import librosa
import numpy as np
from ffmpy import FFmpeg
import matplotlib.pyplot as plt
from scipy.fft import fft
from IPython.display import Audio, display
from playsound import playsound


def read_audio(audio_file, sr=16000):
    """
    Load an audio file as a floating point time series
    :param audio_file:
    :param sr: sampling rate
    :return:
    """
    audio_data, sr = librosa.load(audio_file, sr=sr)
    return audio_data, sr


def save_audio(audio_file, audio_data, sr=16000):
    """
    save audio file
    :param audio_file:
    :param audio_data:
    :param sr: sampling rate
    :return:
    """
    soundfile.write(audio_file, audio_data, samplerate=sr)


def audio_write(audio_file, audio_data: np.ndarray, sr=16000, format="wav", buffer=False):
    """
    保存音频
    :param audio_file: 音频文件,或者
    :param audio_data: 音频数据,shape is (Nums,)
    :param sr: 音频采样率
    :param format:     音频格式
    :return:
    """

    if buffer:
        audio_buffer = io.BytesIO()
        soundfile.write(audio_buffer, audio_data, samplerate=sr, format=format)
        return audio_buffer
    else:
        soundfile.write(audio_file, audio_data, samplerate=sr, format=format)
        return audio_file


def sound_audio(audio_file):
    """
    播放音频
    :param audio_file:
    :return:
    """
    if isinstance(audio_file, str):
        sound_audio_file(audio_file, jupyter=False)
    else:
        sound_audio_buffer(audio_file)


def sound_audio_file(audio_file, jupyter=False):
    """
    :param audio_file:
    :param jupyter:
    :return:
    """
    if jupyter:
        display(Audio(audio_file))
    else:
        playsound(audio_file)


def sound_audio_buffer(audio_buffer, tmp_audio="./tmp.wav"):
    """
    :param audio_buffer:
    :param tmp_audio:
    :return:
    """

    if isinstance(audio_buffer, bytes):
        audio_data, sr = audio_bytes2array(audio_buffer)
    elif isinstance(audio_buffer, io.BytesIO):
        audio_buffer_ = copy.deepcopy(audio_buffer)
        audio_data, sr = soundfile.read(audio_buffer_)
    tmp_audio = audio_write(tmp_audio, audio_data, sr=sr)
    sound_audio_file(tmp_audio)


def audio_bytes2array(bin_data):
    """
    将二进制音频数据解码为原始数组
    :param bin_data:二进制数据
    :return:
    """
    b = io.BytesIO(bin_data)
    data, sample_rate = soundfile.read(b)  # 使用soundfile库读二进制音频文件，data为音频数据，sr为采样率
    return data, sample_rate


def write_bin_file(file, bin_data):
    """
    保存二进制数据(图片,视频,音频等任意二进制的数据文件)
    :param file: 图片,视频,音频等任意二进制的数据文件
    :param bin_data:二进制数据
    :return:
    """
    with open(file, "wb") as w:
        w.write(bin_data)


def extract_video_audio(video_file: str, audio_file: str):
    """
    提取视频声音
    >> os.system(f'ffmpeg -i {video_file} -ac 1 -y {audio_file}')
    :param video_file: 输入视频文件
    :param audio_file: 输出音频文件
    :return:
    """
    if os.path.exists(audio_file): os.remove(audio_file)
    _ext_audio = os.path.basename(audio_file).split(".")[-1]
    if _ext_audio not in ['mp3', 'wav']: raise Exception('audio format not support')
    # ff = FFmpeg(inputs={video_file: None}, outputs={audio_file: '-f {} -vn'.format(_ext_audio)})
    # ff = FFmpeg(inputs={video_file: None}, outputs={audio_file: ' -ac 1 -y'.format(_ext_audio)})
    # ff.run()
    os.system(f'ffmpeg -i {video_file} -ac 1 -y {audio_file}')
    return audio_file


def merge_video_audio(video_file: str, audio_file: str, video_out: str):
    """
    合并视频和语音,实现视频添加音频
    :param video_file: 输入视频文件
    :param audio_file: 输入音频文件
    :param video_out:  输出视频+音频的文件
    :return:
    """
    if os.path.exists(video_out): os.remove(video_out)
    _ext_video = os.path.basename(video_file).strip().split('.')[-1]
    _ext_audio = os.path.basename(audio_file).strip().split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']: raise Exception('audio format not support')
    _codec = 'copy'
    if _ext_audio == 'wav': _codec = 'aac'
    # ff = FFmpeg(inputs={video_file: None, audio_file: None},
    #             outputs={video_out: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    # ff.run()
    # 用合成音频替换原有音频
    os.system(f'ffmpeg -i {video_file} -i {audio_file} -c:v copy -map 0:v:0 -map 1:a:0 -shortest -y {video_out}')
    return video_out


def display_time_domain(audio, sr=16000):
    """
    显示语音时域波形
    https://zhuanlan.zhihu.com/p/371394137
    :param audio:  audio file or audio data
    :param sr : sampling rate
    :return:
    """
    if isinstance(audio, str) and os.path.isfile(audio):
        samples, sr = read_audio(audio, sr=sr)
    else:
        samples = audio
    time = np.arange(0, len(samples)) * (1.0 / sr)
    plt.plot(time, samples)
    plt.title("Time Domain Waveform(sr={})".format(sr))
    plt.xlabel("Time/s")
    plt.ylabel("Amplitude")
    plt.show()


def display_freq_domain(audio, sr=16000):
    """
    显示语音频域波形
    https://zhuanlan.zhihu.com/p/371394137
    :param audio:  audio file or audio data
    :param sr : sampling rate
    :return:
    """
    if isinstance(audio, str) and os.path.isfile(audio):
        samples, sr = read_audio(audio, sr=sr)
    else:
        samples = audio
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)
    ft = fft(samples)
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)
    # plot spectrum，限定[:40000]
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title("Freq Domain Waveform(sr={})".format(sr))
    plt.xlabel("Freq/Hz")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == '__main__':
    # video_file = "../data/video/kunkun_cut.mp4"
    # audio_file = "../data/video/kunkun_cut.mp3"
    # extract_video_audio(video_file, audio_file)
    video_file = "../../data/video/test-video.mp4"
    audio_file = "../../data/video/kunkun_cut.mp3"
    video_out = "../../data/video/test-video-result.mp4"
    audio_file = "../../data/audio/bus_chinese.wav"
    # playsound(audio_file)
    # playsound(audio_file)
    display_time_domain(audio_file)
    display_freq_domain(audio_file)
    # merge_video_audio(video_file, audio_file, video_out)
