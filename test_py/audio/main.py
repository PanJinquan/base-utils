# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail : 
    @Date   : 2023-08-01 22:27:56
    @Brief  :
"""
import cv2
import numpy as np
import librosa


def cv_show_image(title, image, use_rgb=False, delay=0):
    """
    调用OpenCV显示图片
    :param title: 图像标题
    :param image: 输入是否是RGB图像
    :param use_rgb: True:输入image是RGB的图像, False:返输入image是BGR格式的图像
    :param delay: delay=0表示暂停，delay>0表示延时delay毫米
    :return:
    """
    img = image.copy()
    if img.shape[-1] == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    # cv2.namedWindow(title, flags=cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(delay)
    return img


def librosa_feature_melspectrogram(y,
                                   sr=16000,
                                   n_mels=128,
                                   n_fft=2048,
                                   hop_length=256,
                                   win_length=None,
                                   window="hann",
                                   center=True,
                                   pad_mode="reflect",
                                   power=2.0,
                                   fmin=0.0,
                                   fmax=None,
                                   **kwargs):
    """
    计算音频梅尔频谱图(Mel Spectrogram)
    :param y: 音频时间序列
    :param sr: 采样率
    :param n_mels: number of Mel bands to generate产生的梅尔带数
    :param n_fft:  length of the FFT window FFT窗口的长度
    :param hop_length: number of samples between successive frames 帧移(相邻窗之间的距离)
    :param win_length: 窗口的长度为win_length，默认win_length = n_fft
    :param window:
    :param center: 如果为True，则填充信号y，以使帧 t以y [t * hop_length]为中心。
                   如果为False，则帧t从y [t * hop_length]开始
    :param pad_mode:
    :param power: 幅度谱的指数。例如1代表能量，2代表功率，等等
    :param fmin: 最低频率（Hz）
    :param fmax: 最高频率(以Hz为单位),如果为None,则使用fmax = sr / 2.0
    :param kwargs:
    :return: 返回Mel频谱shape=(n_mels,n_frames),n_mels是Mel频率的维度(频域),n_frames为时间帧长度(时域)
    """
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         S=None,
                                         n_mels=n_mels,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         window=window,
                                         center=center,
                                         pad_mode=pad_mode,
                                         power=power,
                                         fmin=fmin,
                                         fmax=fmax,
                                         **kwargs)
    return mel


def librosa_feature_mfcc(y,
                         sr=16000,
                         n_mfcc=128,
                         n_mels=128,
                         n_fft=2048,
                         hop_length=256,
                         win_length=None,
                         window="hann",
                         center=True,
                         pad_mode="reflect",
                         power=2.0,
                         fmin=0.0,
                         fmax=None,
                         dct_type=2,
                         **kwargs):
    """
    计算音频MFCC
    :param y: 音频时间序列
    :param sr: 采样率
    :param n_mfcc: number of MFCCs to return
    :param n_mels: number of Mel bands to generate产生的梅尔带数
    :param n_fft:  length of the FFT window FFT窗口的长度
    :param hop_length: number of samples between successive frames 帧移(相邻窗之间的距离)
    :param win_length: 窗口的长度为win_length，默认win_length = n_fft
    :param window:
    :param center: 如果为True，则填充信号y，以使帧 t以y [t * hop_length]为中心。
                   如果为False，则帧t从y [t * hop_length]开始
    :param pad_mode:
    :param power: 幅度谱的指数。例如1代表能量，2代表功率，等等
    :param fmin: 最低频率（Hz）
    :param fmax: 最高频率(以Hz为单位),如果为None,则使用fmax = sr / 2.0
    :param kwargs:
    :return: 返回MFCC shape=(n_mfcc,n_frames)
    """
    # MFCC 梅尔频率倒谱系数
    mfcc = librosa.feature.mfcc(y=y,
                                sr=sr,
                                S=None,
                                n_mfcc=n_mfcc,
                                n_mels=n_mels,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                window=window,
                                center=center,
                                pad_mode=pad_mode,
                                power=power,
                                fmin=fmin,
                                fmax=fmax,
                                dct_type=dct_type,
                                **kwargs)
    return mfcc


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


def print_feature(name, feature):
    h, w = feature.shape[:2]
    np.set_printoptions(precision=7, suppress=True, linewidth=(11 + 3) * w)
    print("------------------------{}------------------------".format(name))
    for i in range(w):
        v = feature[:, i].reshape(-1)
        print("data[{:0=3d},:]={}".format(i, v))


def print_vector(name, data):
    np.set_printoptions(precision=7, suppress=False)
    print("------------------------%s------------------------\n" % name)
    print("{}".format(data.tolist()))


if __name__ == '__main__':
    sr = None
    n_fft = 400
    hop_length = 160
    n_mel = 64
    fmin = 80
    fmax = 7600
    n_mfcc = 64
    dct_type = 2
    power = 2.0
    center = False
    norm = True
    window = "hann"
    pad_mode = "reflect"
    # audio_file = "data/data_s2.wav"
    audio_file = "/media/PKing/新加卷1/SDK/audio/Librosa-Cpp/data/data_s1.wav"
    data, sr = read_audio(audio_file, sr=sr, mono=False)
    print("n_fft      = %d" % n_fft)
    print("n_mel      = %d" % n_mel)
    print("hop_length = %d" % hop_length)
    print("fmin, fmax = (%d,%d)" % (fmin, fmax))
    print("sr         = %d, data size=%d" % (sr, len(data)))
    # print_vector("audio data", data)
    mels_feature = librosa_feature_melspectrogram(y=data,
                                                  sr=sr,
                                                  n_mels=n_mel,
                                                  n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  win_length=None,
                                                  fmin=fmin,
                                                  fmax=fmax,
                                                  window=window,
                                                  center=center,
                                                  pad_mode=pad_mode,
                                                  power=power)
    print_feature("mels_feature", mels_feature)
    print("mels_feature size(n_frames,n_mels)=({},{})".format(mels_feature.shape[1], mels_feature.shape[0]))
    cv_show_image("mels_feature(Python)", mels_feature, delay=10)

    mfcc_feature = librosa_feature_mfcc(y=data,
                                        sr=sr,
                                        n_mfcc=n_mfcc,
                                        n_mels=n_mel,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        win_length=None,
                                        fmin=fmin,
                                        fmax=fmax,
                                        window=window,
                                        center=center,
                                        pad_mode=pad_mode,
                                        power=power,
                                        dct_type=dct_type)
    print_feature("mfcc_feature", mfcc_feature)
    print("mfcc_feature size(n_frames,n_mfcc)=({},{})".format(mfcc_feature.shape[1], mfcc_feature.shape[0]))
    cv_show_image("mfcc_feature(Python)", mfcc_feature, delay=10)

    cv2.waitKey(0)
