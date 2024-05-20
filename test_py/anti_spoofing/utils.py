# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-05-20 17:18:13
    @Brief  :
"""
import os
import cv2
import argparse
from datetime import datetime


class StereoCamera(object):
    def __init__(self, rgb_video, inf_video, width=640, height=480):
        """采集RGB和红外视频数据"""
        self.width = width
        self.height = height
        self.cap_rgb = get_video_capture(rgb_video, width=width, height=height)
        self.cap_inf = get_video_capture(inf_video, width=width, height=height)
        self.count = 0

    def capture(self, out_root, name, prefix="paper"):
        """
        数据类型：prefix = ["real","paper(纸)","monitor(显示屏)","mask(面具)"]
        :return:
        """
        width, height, num_frames, fps = get_video_info(self.cap_rgb)
        width, height, num_frames, fps = get_video_info(self.cap_inf)
        video_name = get_time()
        out_dir = os.path.join(out_root, "video", name)
        save_rgb = os.path.join(out_dir, "{}_{}_{}_rgb.avi".format(name, prefix, video_name))
        save_inf = os.path.join(out_dir, "{}_{}_{}_inf.avi".format(name, prefix, video_name))
        writer_rgb = get_video_writer(save_rgb, width=width, height=height, fps=fps)
        writer_inf = get_video_writer(save_inf, width=width, height=height, fps=fps)
        self.count = 0
        while True:
            rgb_ok, rgb_frame = self.cap_rgb.read()
            inf_ok, inf_frame = self.cap_inf.read()
            if not (rgb_ok and inf_ok):
                msg = "打开摄像头失败，请检查USB连接" if self.count == 0 else "视频已结束"
                print(msg)
                break
            self.count += 1
            self.task(rgb_frame, inf_frame)
            writer_rgb.write(rgb_frame)
            writer_inf.write(inf_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
                break
        writer_rgb.release()
        writer_inf.release()
        self.cap_rgb.release()
        self.cap_inf.release()
        cv2.destroyAllWindows()

    def task(self, rgb_frame, inf_frame, delay=5):
        cv_show_image("rgb_frame", rgb_frame, delay=5)
        cv_show_image("inf_frame", inf_frame, delay=delay)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


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


def get_time(format="s"):
    """
    :param format:
    :return:
    """
    if format.lower() == "s":
        # time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    elif format.lower() == "p":
        # 20200508_143059_379116
        time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')
        time = time[:-2]
    else:
        time = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
    return time


def get_video_capture(video, width=None, height=None, fps=None):
    """
     获得视频读取对象
     --   7W   Pix--> width=320,height=240
     --   30W  Pix--> width=640,height=480
     720P,100W Pix--> width=1280,height=720
     960P,130W Pix--> width=1280,height=1024
    1080P,200W Pix--> width=1920,height=1080
    :param video: video file or Camera ID
    :param width:   图像分辨率width
    :param height:  图像分辨率height
    :param fps:  设置视频播放帧率
    :return:
    """
    if len(video) == 1: video = int(video)
    video_cap = cv2.VideoCapture(video)
    # 设置分辨率
    if width:
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        video_cap.set(cv2.CAP_PROP_FPS, fps)
    return video_cap


def get_video_writer(video_file, width, height, fps):
    """
    获得视频存储对象
    :param video_file: 输出保存视频的文件路径
    :param width:   图像分辨率width
    :param height:  图像分辨率height
    :param fps:  设置视频播放帧率
    :return:
    """
    if not os.path.exists(os.path.dirname(video_file)):
        os.makedirs(os.path.dirname(video_file))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameSize = (int(width), int(height))
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, frameSize)
    print("save video:{},width:{},height:{},fps:{}".format(video_file, width, height, fps))
    return video_writer


def get_video_info(video_cap: cv2.VideoCapture):
    """
    获得视频的基础信息
    :param video_cap:视频对象
    :return:
    """
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    print("read video:width:{},height:{},fps:{},num_frames:{}".format(width, height, fps, num_frames))
    return width, height, num_frames, fps
