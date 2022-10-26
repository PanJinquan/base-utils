# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-05-24 16:46:51
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


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
    video_cap = cv2.VideoCapture(video)
    # 设置分辨率
    if width:
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        video_cap.set(cv2.CAP_PROP_FPS, fps)
    return video_cap


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
    print("video:width:{},height:{},fps:{},num_frames:{}".format(width, height, fps, num_frames))
    return width, height, num_frames, fps


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
    print("video:width:{},height:{},fps:{}".format(width, height, fps))
    return video_writer


def video2gif(video_file, gif_file=None, func=None, interval=1, use_pil=False, fps=-1, vis=True):
    """
    将视频文件直接转为GIF图像
    :param video_file: 输入视频文件
    :param gif_file: 保存GIF图文件
    :param func:
    :param interval:
    :param use_pil: True使用PIL库生成GIF图，文件小，但质量较差
                    False使用imageio库生成GIF图，文件大，但质量较好
    :param vis:
    :return:
    """
    name = os.path.basename(video_file).split(".")[0]
    if not gif_file:
        gif_file = os.path.join(os.path.dirname(video_file), name + ".gif")
    video_cap = get_video_capture(video_file)
    width, height, num_frames, _fps = get_video_info(video_cap)
    if not os.path.exists(gif_file): file_utils.create_file_path(gif_file)
    count = 0
    frames = []
    while True:
        if count % interval == 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if func:
                frame = func(frame)
            if vis:
                image_utils.cv_show_image("frame", frame, use_rgb=False, delay=30)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    video_cap.release()
    fps = _fps / interval if fps <= 0 else fps
    if use_pil:
        image_utils.frames2gif_by_pil(frames, gif_file=gif_file, fps=fps, loop=0)
    else:
        image_utils.frames2gif_by_imageio(frames, gif_file=gif_file, fps=fps, loop=0)


def video2frames(video_file, out_dir=None, func=None, interval=1, vis=True):
    """
    视频抽帧图像
    :param video_file: 视频文件
    :param out_dir: 保存抽帧图像的目录
    :param func: 回调函数，对每一帧图像进行处理
    :param interval: 保存间隔
    :param vis: 是否可视化显示
    :return:
    """
    name = os.path.basename(video_file).split(".")[0]
    if not out_dir:  out_dir = os.path.join(os.path.dirname(video_file), name)
    video_cap = get_video_capture(video_file)
    width, height, num_frames, fps = get_video_info(video_cap)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    while True:
        if count % interval == 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if func:
                frame = func(frame)
            if vis:
                image_utils.cv_show_image("frame", frame, use_rgb=False, delay=30)
            frame_file = os.path.join(out_dir, "{}_{:0=4d}.jpg".format(name, count))
            cv2.imwrite(frame_file, frame)
        count += 1
    video_cap.release()


def frames2video(image_dir, video_file=None, func=None, size=None, postfix=["*.png", "*.jpg"], interval=1, fps=30,
                 vis=True):
    """
    将抽帧图像转为视频文件(*.mp4)
    :param image_dir:抽帧图像路径
    :param video_file: 保存的视频文件
    :param func: 回调函数，对每一帧图像进行处理
    :param postfix:
    :param interval:
    :param fps:
    :param vis:
    :return:
    """
    if isinstance(image_dir, list):
        image_list = image_dir
    else:
        image_list = file_utils.get_files_list(image_dir, postfix=postfix)
        image_list = sorted(image_list)
    if not video_file:
        video_file = os.path.join(image_dir) + "_{}.mp4".format(file_utils.get_time('p'))
    if len(image_list) == 0: return
    if not size:
        height, width = cv2.imread(image_list[0]).shape[:2]
    else:
        width, height = size
    video_writer = get_video_writer(video_file, width, height, fps)
    for count, image_file in tqdm(enumerate(image_list)):
        if count % interval == 0:
            frame = cv2.imread(image_file)
            if func:
                frame = func(frame)
            else:
                frame = image_utils.resize_image_padding(frame, size=(width, height))
            if vis:
                image_utils.cv_show_image("frame", frame, use_rgb=False, delay=30)
            video_writer.write(frame)
    video_writer.release()


def convert_video_format(video_file, save_video, interval=1):
    return video2video(video_file, save_video, interval=interval)


def video2video(video_file, save_video, interval=1, vis=True, delay=20):
    """
    转换视频格式
    :param video_file: *.avi,*.mp4,...
    :param save_video: *.avi
    :param interval: 间隔
    :return:
    """
    video_cap = get_video_capture(video_file)
    width, height, num_frames, fps = get_video_info(video_cap)
    video_writer = get_video_writer(save_video, width, height, fps)
    # freq = int(fps / detect_freq)
    count = 0
    while True:
        if count % interval == 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if vis: image_utils.cv_show_image("frame", frame, use_rgb=False, delay=delay)
            video_writer.write(frame)
        count += 1
    video_cap.release()


def write_video(self, frame):
    self.video_writer.write(frame)


class CVVideo():
    def __init__(self):
        pass

    def start_capture(self, video_file, save_video=None, interval=1):
        """
        start capture video
        :param video_file: *.avi,*.mp4,...
        :param save_video: *.avi
        :param interval:
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
        video_cap = get_video_capture(video_file)
        width, height, num_frames, fps = get_video_info(video_cap)
        if save_video:
            self.video_writer = get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            if count % interval == 0:
                # 设置抽帧的位置
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                isSuccess, frame = video_cap.read()
                if not isSuccess:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.task(frame)
            if save_video:
                self.write_video(frame)
            count += 1
        video_cap.release()

    def write_video(self, frame):
        self.video_writer.write(frame)

    def task(self, frame):
        # TODO
        cv2.imshow("image", frame)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(10)
        return frame


def target_task(frame):
    frame = image_utils.resize_image(frame, size=(960, None))
    return frame


if __name__ == "__main__":
    video_file = "/home/dm/nasdata/release/CSDN/双目测距Demo视频(Python).MP4"
    # video_file = "/home/dm/视频/双目测距Demo视频(Python).mp4"
    # dst_file = "/home/dm/视频/双目测距Demo视频(Python)1.mp4"
    # video2frames(video_file, interval=10, vis=True)
    # frames2video(image_dir, interval=1, vis=True)
    video2gif(video_file, interval=15, func=target_task, fps=3,use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)
