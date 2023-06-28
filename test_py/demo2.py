# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils
from pybaseutils.base_audio import audio_utils
import socket
import cv2
import numpy
from time import sleep



if __name__ == '__main__':
    # video_file = "../data/video/kunkun_cut.mp4"
    # audio_file = "../data/video/kunkun_cut.mp3"
    # extract_video_audio(video_file, audio_file)
    video_file = "/home/PKing/Downloads/xmc-video.ts"
    audio_file = "/home/PKing/nasdata/Project/Avatar/data/轮_zh-CN-XiaoshuangNeural.wav"
    video_out = "/home/PKing/Downloads/xmc-video-result.mp4"
    # audio_utils.playsound(audio_file)
    audio_utils.merge_video_audio(video_file, audio_file, video_out)
