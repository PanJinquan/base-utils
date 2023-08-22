# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import asyncio
import time
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils, json_utils
from pybaseutils.audio import audio_utils

video_file = "/home/PKing/Downloads/audio/data/主视角-1号位-完整.mp4"
video_file = "http://10.13.3.5:8088/camera1/主视角-1号位-完整.mp4"
audio_file = "/home/PKing/nasdata/dataset-dmai/AIJE/audio/test-audio/test-1号视角.flv"
# audio_file = "/home/PKing/Downloads/test-1号视角.flv"
# audio_file = "/home/PKing/Downloads/主视角-1号位-完整.mp4"
# audio_file = "/home/PKing/Downloads/1.flv"
# audio_file = "/home/PKing/Downloads/audio/data/主视角-1号位-完整-16k-s16le.wav"
audio_utils.extract_video_audio(audio_file)
# ffmpeg -re -i http://10.13.3.5:8088/camera1/主视角-1号位-完整.mp4 -vn -ar 16000 -ac 1 -f s16le
# os.system(f'ffmpeg -i {video_file} -ac 1 -y {audio_file}')
# os.system(f'ffmpeg -i {video_file} -ac 1 -y -ar 16000 {audio_file}')
# os.system(f'ffmpeg -re -i {video_file} -vn -ar 16000 -ac 1 -f s16le -y {audio_file}')
# os.system(f'ffmpeg -i {video_file} -vn -ar 16000 -ac 1 -f s16le -y {audio_file}')
# os.system(f'ffmpeg -i {video_file} -vn -ar 16000 -ac 1 -y {audio_file}')
