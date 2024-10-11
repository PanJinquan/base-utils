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
import soundfile as sf
import copy
import librosa
import wave
import numpy as np
from ffmpy import FFmpeg
import matplotlib.pyplot as plt
from scipy.fft import fft
from IPython.display import Audio, display
from playsound import playsound
from scipy.io import wavfile

import numpy as np


def load_pcm(filename):
    with open(filename, 'rb') as f:  data = f.read()
    return data


def float32toint16(data_f32):
    omin = -32768
    omax = 32767
    imin = -1.0
    imax = 1.0
    # 将数据float32从[-1.0,1.0]映射到int16
    data_int16 = (omax - omin) * (data_f32 - imin) / (imax - imin) + omin
    data_int16 = np.array(data_int16, dtype=np.int16)
    return data_int16


def int16tofloat32(data_int16):
    imin = -32768
    imax = 32767
    omin = -1.0
    omax = 1.0
    audio_data = np.array(data_int16, dtype=np.float32)
    # 将数据int16映射到[-1.0,1.0] float32
    data_f32 = (omax - omin) * (audio_data - imin) / (imax - imin) + omin
    return data_f32


def audio_data2pcm_bytes(audio_data: np.ndarray, data_type=np.int16):
    """
    将语音数据转换PCM数据
    :param audio_data:  np.ndarray，audio time series，语音数据，只支持单声道
                        一般使用librosa.load()直接读取即可
                        audio_data, sr = librosa.load(audio_file, sr=sr, mono=mono)
    :param data_type: 数据类型
    :return:
    """
    data = float32toint16(audio_data)
    bytes = data.tobytes()  # array转换为bytes
    return bytes


def pcm_bytes2audio_data(pcm_bytes, data_type=np.int16):
    """
    将PCM数据转为语音数据
    :param pcm_bytes: PCM数据(int16)
    :param data_type: 数据类型
    :return:
    """
    audio_data = np.frombuffer(pcm_bytes, dtype=data_type)
    audio_data = int16tofloat32(audio_data)
    return audio_data


def pcm2wav(pcm_file, wav_file="", channels=1, bits=16, sr=16000):
    """
    利用wave库，添加通道信息、采样位数、采样率等信息作为文件头，pcm数据直接写入即可。
    :param pcm_file:
    :param wav_file:
    :param channels:
    :param bits:
    :param sr:
    :return:
    """
    if not wav_file: wav_file = pcm_file.replace(".pcm", ".wav")
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()
    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))
    wavfile = wave.open(wav_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)  # #16位采样即为2字节
    wavfile.setframerate(sr)
    wavfile.writeframes(pcmdata)
    wavfile.close()


def wav2pcm(wav_file, pcm_file, data_type=np.int16):
    """
    pcm转wav(支持多声道和单声道)：https://www.45fan.com/article.php?aid=1Hzw69oiQl7fA1VM
    :param wav_file:
    :param pcm_file:
    :param data_type:
    :return:
    """
    if not pcm_file: pcm_file = wav_file.replace(".wav", ".pcm")
    wf = wave.open(wav_file, "rb")
    params = wf.getparams()
    channels, bits_w, sr, nums = params[:4]
    wf.close()
    # 将文件头去掉，数据转成int16型即可
    f = open(wav_file, "rb")
    f.seek(0)
    f.read(44)
    data = np.fromfile(f, dtype=data_type)
    data.tofile(pcm_file)
    return sr, channels


def wav2pcm_mono(wav_file, pcm_file="", data_type=np.int16):
    """
    pcm转wav(仅支持单声道)将wav文件头去掉，数据转成int型即为pcm
    :param wav_file:
    :param pcm_file:
    :param data_type:
    :return:
    """
    if not pcm_file: pcm_file = wav_file.replace(".wav", ".pcm")
    audio_data, sr = read_audio(wav_file, sr=16000, mono=True)
    channels = 1
    save_pcm_from_numpy(audio_data, pcm_file, data_type=data_type)
    return sr, channels


def save_pcm_from_numpy(audio_data: np.ndarray, pcm_file, data_type=np.int16):
    """
    将numpy格式的语音数据保存为PCM文件
    :param audio_data:  float32数据[-1.0,1.0],只支持单声道
    :param pcm_file: 保存PCM文件
    :param data_type: PCM数据类型，一般为int16
    :return:
    """
    omin = -32768
    omax = 32767
    imin = -1.0
    imax = 1.0
    # 将数据float32转换到int16
    audio_data = (omax - omin) * (audio_data - imin) / (imax - imin) + omin
    data = np.array(audio_data, dtype=data_type)
    data.tofile(pcm_file)
    bytes = data.tobytes()  # array转换为bytes
    return bytes


def librosa_load(audio_file, sr=None, mono=True):
    """
    默认将多声道音频文件转换为单声道，并返回一维数组；
    如果你需要处理多声道音频文件，可以使用 mono=False,参数来保留所有声道，并返回二维数组。
    :param audio_file:
    :param sr: sampling rate 16000
    :param mono: 设置为true是单通道，否则是双通道
    :return:
    """
    audio_data, sr = librosa.load(audio_file, sr=sr, mono=mono)  # 非常耗时
    return audio_data, sr


def soundfile_load(audio_file, sr=None, mono=True):
    """
    默认会将多声道音频文件的每个声道分别存储在二维数组的不同列中，因此返回的是一个二维数组
    :param audio_file:
    :param sr: sampling rate 当设置的采样率与音频原始采样率不一致时，会进行重采样，导致非常耗时
    :param mono: 设置为true是单通道，否则是双通道
    :return:
    """
    audio_data, orig_sr = sf.read(audio_file, samplerate=None, dtype="float32")  # 不能直接修改samplerate
    if mono and audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=-1)
    if sr is None or orig_sr == sr:
        sr = orig_sr
    else:
        audio_data = librosa.resample(y=audio_data, orig_sr=orig_sr, target_sr=sr)  # 使用librosa进行重采样至目标采样率
    return audio_data, sr


def wavfile_load(audio_file):
    sr, wav_data = wavfile.read(audio_file)  # int16类型
    # 转为float64类型
    wav_data = wav_data / (32768)
    # wav_data:[-0.03305054 -0.03561401 -0.038114697]
    return wav_data, sr


def read_audio(audio_file, sr=16000, mono=True):
    """
    默认将多声道音频文件转换为单声道，并返回一维数组；
    如果你需要处理多声道音频文件，可以使用 mono=False,参数来保留所有声道，并返回二维数组。
    :param audio_file:
    :param sr: sampling rate
    :param mono: 设置为true是单通道，否则是双通道
    :return:
    """
    audio_data, sr = librosa_load(audio_file, sr=sr, mono=mono)  # 巨慢
    # audio_data, sr = soundfile_load(audio_file, sr=sr, mono=mono)  # 巨慢
    return audio_data, sr


def save_audio(audio_file, audio_data, sr=16000):
    """
    save audio file
    :param audio_file:
    :param audio_data:
    :param sr: sampling rate
    :return:
    """
    sf.write(audio_file, audio_data, samplerate=sr)


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
        sf.write(audio_buffer, audio_data, samplerate=sr, format=format)
        return audio_buffer
    else:
        sf.write(audio_file, audio_data, samplerate=sr, format=format)
        return audio_file


def sound_audio(audio, sr=16000):
    """
    播放音频
    :param audio: ndarray or string
    :return:
    """
    if isinstance(audio, np.ndarray):
        sound_audio_data(audio, sr=sr)
    elif isinstance(audio, str):
        sound_audio_file(audio, jupyter=False)
    else:
        sound_audio_buffer(audio)


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
        audio_data, sr = sf.read(audio_buffer_)
    tmp_audio = audio_write(tmp_audio, audio_data, sr=sr)
    sound_audio_file(tmp_audio)


def sound_audio_data(audio_data: np.ndarray, sr=16000, tmp_audio="./tmp.wav"):
    """
    播放声音
    :param audio_data:
    :param sr:
    :return:
    """
    tmp_audio = audio_write(tmp_audio, audio_data, sr=sr)
    sound_audio_file(tmp_audio)


def audio_bytes2array(bin_data):
    """
    将二进制音频数据解码为原始数组
    :param bin_data:二进制数据
    :return:
    """
    b = io.BytesIO(bin_data)
    data, sample_rate = sf.read(b)  # 使用soundfile库读二进制音频文件，data为音频数据，sr为采样率
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


def get_feature_melspectrogram(wav, sr, n_fft=128, hop_length=256):
    """
    计算音频梅尔频谱图(Mel Spectrogram)
    :param wav:
    :param sr:
    :return:
    """
    # Mel Spectrogram 梅尔频谱图
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    # mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    return mel


def get_feature_mfcc(wav, sr, n_mfcc=128, hop_length=256):
    """
    计算音频信号的MFCC:
    先调用melspectrogram，计算梅尔频谱，然后取对数:power_to_db
    参考：mfcc = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))
    :param wav:
    :param sr:
    :return:
    """
    # MFCC 梅尔频率倒谱系数
    # mfcc = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, htk=True)
    return mfcc


def extract_video_audio(video_file: str, audio_file: str = ""):
    """
    提取视频声音
    ffmpeg -i {file_path} -f wav -ar 16000 {file_name}file_path:
    视频的文件路径
    file_name: 文件名称
    -ar: 设置音频采样频率。对于输出流，它默认设置为相应输入流的频率。对于输入流，此选项仅对音频抓取设备和原始解复用器有意义，并映射到相应的解复用器选项
    -i: 输入文件网址
    -f: 强制输入或输出文件格式。通常会自动检测输入文件的格式，并根据输出文件的文件扩展名猜测格式，因此在大多数情况下不需要此选项。
    >> os.system(f'ffmpeg -i {video_file} -ac 1 -y {audio_file}')
    :param video_file: 输入视频文件
    :param audio_file: 输出音频文件
    :return:
    """
    if not audio_file: audio_file = video_file[:-len(".mp4")] + ".wav"
    if os.path.exists(audio_file): os.remove(audio_file)
    _ext_audio = os.path.basename(audio_file).split(".")[-1]
    # if _ext_audio not in ['mp3', 'wav']: raise Exception('audio format not support')
    # ff = FFmpeg(inputs={video_file: None}, outputs={audio_file: '-f {} -vn -ac 1 -ar 16000'.format(_ext_audio)})
    # ff = FFmpeg(inputs={video_file: None}, outputs={audio_file: ' -ac 1 -y'.format(_ext_audio)})
    # ff.run()
    os.system(f'ffmpeg -i {video_file} -ac 1 -y -ar 16000 {audio_file}')
    # os.system(f'ffmpeg -i {video_file} -ac 1 -y {audio_file}')
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
    length = samples.shape[0] / sr
    time = np.arange(0, len(samples)) * (1.0 / sr)
    plt.plot(time, samples)
    plt.title(f"Time Domain Waveform(sr={sr},t={length}")
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
    length = samples.shape[0] / sr
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)
    ft = fft(samples)
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)
    # plot spectrum，限定[:40000]
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title(f"Freq Domain Waveform(sr={sr},t={length}")
    plt.xlabel("Freq/Hz")
    plt.ylabel("Amplitude")
    plt.show()


def test_time(audio_file, sr=None):
    from pybaseutils import time_utils
    for i in range(10):
        with time_utils.Performance("soundfile_load") as p:
            audio_data1, sr1 = soundfile_load(audio_file, sr=sr)
        with time_utils.Performance("librosa_load") as p:
            audio_data2, sr2 = librosa_load(audio_file, sr=sr)
    print(np.sum(np.abs(audio_data2 - audio_data1)))


if __name__ == '__main__':
    # video_file = "../data/video/kunkun_cut.mp4"
    # audio_file = "../data/video/kunkun_cut.mp3"
    # extract_video_audio(video_file, audio_file)
    video_file = "../../data/video/test-video.mp4"
    audio_file = "../../data/video/kunkun_cut.mp3"
    video_out = "../../data/video/test-video-result.mp4"
    audio_file = "../../data/audio/bus_chinese.wav"

    test_time(audio_file)
    # playsound(audio_file)
    # playsound(audio_file)
    # display_time_domain(audio_file)
    # display_freq_domain(audio_file)
    # merge_video_audio(video_file, audio_file, video_out)
