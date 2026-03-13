# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import os
import cv2
import random
import types
import torch
import numbers
from typing import List, Tuple, Dict
import numpy as np
from typing import Callable
from PIL import Image
from datetime import datetime
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils, json_utils, base64_utils
from pybaseutils.cvutils import video_utils
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import time
import numpy as np
import threading
import matplotlib


def video_capture(video):
    w, h, num_frames, fps = image_utils.get_video_info(video)
    fpm = 1 / fps
    video_cap = cv2.VideoCapture(video)
    count = 0
    while True:
        print("报告帧率:", video_cap.get(cv2.CAP_PROP_FPS))  # 注意：此值常不准确
        t1 = time.time()
        ret, frame = video_cap.read()
        count += 1
        td = time.time() - t1
        if fpm - td > 0: time.sleep(fpm - td)
        print('count={:0=5d} td={:3.4f}ms'.format(count, td * 1000))


def video_iterator(video):
    w, h, num_frames, fps = image_utils.get_video_info(video)
    video_cap = video_utils.video_iterator(video, save_video=None, vis=False)
    fpm = 1 / fps
    print("fps:", fps)
    print("fpm:", fpm)
    print("num_frames:", num_frames)
    for info in video_cap:
        t1 = info['systime']
        frame = info["frame"]
        count = info['count']
        image_utils.show_image('image', image=frame, delay=1)
        td = time.time() - t1  # 秒
        time.sleep(max(0.0, fpm - td))  # 保证视频有40ms延时
        print('count={:0=5d} td={:3.4f}ms,ts={:3.4f}ms'.format(count, td * 1000, (time.time() - t1) * 1000))


if __name__ == '__main__':
    x = 123
    x = None
    print(f"{str(x):7}")
