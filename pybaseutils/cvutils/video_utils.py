# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-05-24 16:46:51
    @Brief  :
"""
import os
import cv2
import numpy as np
from typing import Callable
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import monitor

VIDEO_POSTFIX = ['*.mp4', '*.avi', '*.mov', "*.flv"]

get_video_capture = image_utils.get_video_capture
get_video_info = image_utils.get_video_info
get_video_writer = image_utils.get_video_writer


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
        if count % interval == 0 and count >= 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess or 0 < num_frames < count: break
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


def video2frames(video_file, out_dir=None, func=None, interval=1, vis=True, delay=10):
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
    if not interval: interval = fps
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    while True:
        if count % interval == 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess or 0 < num_frames < count: break
            if func:
                frame = func(frame)
            if vis:
                image_utils.cv_show_image("frame", frame, use_rgb=False, delay=delay)
            frame_file = os.path.join(out_dir, "{}_{:0=4d}.jpg".format(name, count))
            cv2.imwrite(frame_file, frame)
        count += 1
    video_cap.release()
    cv2.destroyAllWindows()


def video2frames_similarity(video_file, out_dir=None, func=None, interval=1, thresh=0.3, vis=True):
    """
    视频抽帧图像
    :param video_file: 视频文件
    :param out_dir: 保存抽帧图像的目录
    :param func: 回调函数，对每一帧图像进行处理
    :param interval: 保存间隔
    :param vis: 是否可视化显示
    :return:
    """
    sm = monitor.StatusMonitor()
    name = os.path.basename(video_file).split(".")[0]
    if not out_dir:  out_dir = os.path.join(os.path.dirname(video_file), name)
    video_cap = get_video_capture(video_file)
    width, height, num_frames, fps = get_video_info(video_cap)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    last_frame = None
    while True:
        if count % interval == 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, curr_frame = video_cap.read()
            if not isSuccess or 0 < num_frames < count: break
            if func: curr_frame = func(curr_frame)
            if last_frame is None:
                last_frame = curr_frame.copy()
            diff = sm.get_frame_similarity(curr_frame, last_frame, size=(256, 256), vis=False)
            if diff > thresh:
                frame_file = os.path.join(out_dir, "{}_{:0=4d}.jpg".format(name, count))
                last_frame = curr_frame.copy()
                cv2.imwrite(frame_file, curr_frame)
            if vis:
                text = "TH={},diff={:3.3f}".format(thresh, diff)
                image = image_utils.draw_text(curr_frame, point=(10, 70), color=(0, 255, 0),
                                              text=text, drawType="simple")
                image = image_utils.cv_show_image("image", image, delay=5)
        count += 1
    video_cap.release()
    cv2.destroyAllWindows()


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
    cv2.destroyAllWindows()


def video2video(video_file: int or str, save_video: str or int, interval=1, task: Callable = None,
                vis=True, **kwargs):
    """
    转换视频格式
    :param video_file: *.avi,*.mp4,...
    :param save_video: *.avi
    :param interval: 间隔
    :return:
    """
    video_capture(video_file=video_file, save_video=save_video, interval=interval, task=task, vis=vis, **kwargs)


convert_video_format = video2video


def resize_video(video_file, save_video, size=(), start=0, interval=1, vis=True, delay=20):
    """
    转换视频格式
    :param video_file: *.avi,*.mp4,...
    :param save_video: *.avi
    :param interval: 间隔
    :param start:
    :return:
    """
    video_cap = get_video_capture(video_file)
    width, height, num_frames, fps = get_video_info(video_cap)
    frame = np.zeros(shape=(height, width), dtype=np.uint8)
    frame = image_utils.resize_image(frame, size=size)
    height, width = frame.shape[:2]
    video_writer = get_video_writer(save_video, width, height, fps)
    # freq = int(fps / detect_freq)
    count = start
    while True:
        # if count % interval == 0:
        if count % interval == 0 and count > 0:
            # 设置抽帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            if not isSuccess or 0 < num_frames < count: break
            frame = image_utils.resize_image(frame, size=size)
            if vis: image_utils.cv_show_image("frame", frame, use_rgb=False, delay=delay)
            video_writer.write(frame)
        count += 1
    video_cap.release()
    video_writer.release()


def video_capture(video_file: int or str, save_video: str or int = None, interval=1, task: Callable = None,
                  vis=True, **kwargs):
    """
    读取摄像头或者视频流
    :param video_file: String 视频文件，如*.avi,*.mp4,...
                       Int 摄像头ID，如0，1，2
    :param save_video: 保存task视频处理后的结果
    :param interval: 抽帧处理间隔
    :param task: 回调函数： def task(frame, **kwargs)
    :param kwargs:回调函数输入参数,
                 delay: 控制显示延时
                 title: 控制显示窗口名
    :return:
    """
    video_cap = image_utils.get_video_capture(video_file)
    w, h, num_frames, fps = image_utils.get_video_info(video_cap)
    start = int(kwargs.get("start", 0) * fps)
    end = int(kwargs.get("end", num_frames / fps) * fps) if fps > 0 else 0
    end = min(end, num_frames)  # TODO 当num_frames<0时，使用0<end<count继续播放
    interval = fps if interval == -1 else interval  # 当interval=-1，表示interval=fps,即一秒一帧
    save_fps = max(kwargs.get("speed", 1) * fps // interval, 1)
    count = 0
    video_writer = None
    while True:
        ret, frame = video_cap.read()
        if count % interval == 0 and count >= start:
            # TODO 设置抽帧的位置，但某些格式视频容易出现问题
            # if isinstance(video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            # ret, frame = video_cap.read()
            if not ret or 0 < end < count or frame is None: break
            if task: frame = task(frame, **kwargs)
            h, w = frame.shape[:2]
            if vis: image_utils.cv_show_image(kwargs.get("title", "video"), frame, delay=kwargs.get("delay", 10))
            if save_video:
                if not video_writer: video_writer = image_utils.get_video_writer(save_video, w, h, save_fps)
                video_writer.write(frame)
        count += 1
    video_cap.release()
    if video_writer:
        print("save video:{}".format(save_video))
        video_writer.release()


def video_iterator(video_file: int or str, save_video: str or int = None, interval=1, task: Callable = None,
                   vis=False, **kwargs):
    """
    读取摄像头或者视频流迭代器
    Usage:
        from pybaseutils.cvutils import video_utils
        video_cap = video_utils.video_iterator(video_file, save_video, start=4, end=10)
        for data_info in video_cap:
            frame = data_info["frame"]
            ...
    :param video_file: String 视频文件，如*.avi,*.mp4,...
                       Int 摄像头ID，如0，1，2
    :param save_video: 保存task视频处理后的结果
    :param interval: 抽帧处理间隔，当interval=-1，表示当interval=fps,即一秒一帧
    :param task: 回调函数： def task(frame, **kwargs)
    :param kwargs:回调函数输入参数,
                 delay: 控制显示延时,默认10S
                 title: 控制显示窗口名，默认video
                 start: 开始播放时间，单位S
                 end: 结束播放时间，单位S
                 speed: 播放速度
    :return: frame, count, w, h, fps =data_info['frame'],data_info['count'],data_info['w'],data_info['h'],data_info['fps']
    """
    video_cap = image_utils.get_video_capture(video_file)
    w, h, num_frames, fps = image_utils.get_video_info(video_cap)
    start = int(kwargs.get("start", 0) * fps)
    end = int(kwargs.get("end", num_frames / fps) * fps) if fps > 0 else 0
    end = min(end, num_frames)  # TODO 当num_frames<0时，使用0<end<count继续播放
    interval = fps if interval == -1 else interval  # 当interval=-1，表示interval=fps,即一秒一帧
    save_fps = max(kwargs.get("speed", 1) * fps // interval, 1)
    count = 0
    video_writer = None
    while True:
        ret, frame = video_cap.read()
        if count % interval == 0 and count >= start:
            # TODO 设置抽帧的位置，但某些格式视频容易出现问题
            # if isinstance(video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            # ret, frame = video_cap.read()
            if not ret or 0 < end < count or frame is None: break
            if task: frame = task(frame, **kwargs)
            data_info = {"frame": frame, "count": count, "w": w, "h": h, "fps": fps}
            # TODO 返回data_info
            yield data_info
            frame = data_info["frame"]
            h, w = frame.shape[:2]
            if vis: image_utils.cv_show_image(kwargs.get("title", "video"), frame, delay=kwargs.get("delay", 10))
            if save_video:
                if not video_writer: video_writer = image_utils.get_video_writer(save_video, w, h, save_fps)
                video_writer.write(frame)
        count += 1
    video_cap.release()
    if video_writer:
        print("save video:{}".format(save_video))
        video_writer.release()


def resize_task(frame, **kwargs):
    frame = image_utils.resize_image(frame, size=(960, None))
    return frame


def rotation_task(frame, **kwargs):
    count = kwargs.get("count", 0)
    num = 200
    alpha = 0.1
    angle = [-i for i in range(num)][::-1] + [i for i in range(num)]
    angle = angle + angle[::-1]
    h, w = frame.shape[:2]
    frame = image_utils.image_rotation(frame, angle=alpha * angle[count % len(angle)])
    frame = image_utils.get_box_crop(frame, box=[0, 70, w, h - 70])
    return frame


if __name__ == "__main__":
    video_file = "/home/dm/nasdata/release/CSDN/双目测距Demo视频(Python).MP4"
    # video_file = "/home/dm/视频/双目测距Demo视频(Python).mp4"
    # dst_file = "/home/dm/视频/双目测距Demo视频(Python)1.mp4"
    # video2frames(video_file, interval=10, vis=True)
    # frames2video(image_dir, interval=1, vis=True)
    video2gif(video_file, interval=15, func=resize_task, fps=3, use_pil=False, vis=True)
    # video2video(video_file, dst_file, vis=True)
