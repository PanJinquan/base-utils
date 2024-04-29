# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-04 11:13:08
    @Brief  :
"""
import numpy as np
from pybaseutils.audio import audio_utils

if __name__ == '__main__':
    # src_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/test1.wav"
    src_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/bus_chinese.wav"
    # src_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/bus_chinese1.wav"
    pcm_file = "/media/PKing/新加卷1/SDK/base-utils/data/audio/bus.pcm"
    pcm_file2 = "/media/PKing/新加卷1/SDK/base-utils/data/audio/test1.pcm"
    audio_data1, sr = audio_utils.read_audio(audio_file=src_file)
    bytes = audio_utils.audio_data2pcm_bytes(audio_data1)
    audio_data2 = audio_utils.pcm_bytes2audio_data(bytes)
    audio_utils.display_time_domain(audio_data1)
    audio_utils.display_time_domain(audio_data2)
    # print(audio_data1)
    # print(audio_data2)

