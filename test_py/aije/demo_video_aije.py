# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import numpy as np
from xpinyin import Pinyin
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.cvutils import video_utils, monitor

# indoor and outdoor
name_table = {"室内作业": "indoor",
              "室外作业": "outdoor",
              "号视角": "camera"
              }


def video2frames_similarity(video_file, out_dir=None, func=None, interval=1, thresh=0.25, vis=True):
    """
    视频抽帧图像
    :param video_file: 视频文件
    :param out_dir: 保存抽帧图像的目录
    :param func: 回调函数，对每一帧图像进行处理
    :param interval: 保存间隔
    :param thresh: 相似度阈值
    :param vis: 是否可视化显示
    :return:
    """
    sm = monitor.StatusMonitor()
    # 室内
    name = os.path.basename(video_file).split(".")[0]
    name = "{}_{}".format(os.path.basename(os.path.dirname(video_file)), name)
    for k, v in name_table.items(): name = name.replace(k, v)
    # 室外
    # name = file_utils.get_time()
    video_cap = image_utils.get_video_capture(video_file)
    width, height, num_frames, fps = image_utils.get_video_info(video_cap)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    last_frame = None
    while True:
        # 设置抽帧的位置
        # video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        isSuccess, curr_frame = video_cap.read()
        if not isSuccess or 0 < num_frames < count: break
        if count % interval == 0:
            if func: curr_frame = func(curr_frame)
            if last_frame is None:
                last_frame = curr_frame.copy()
            diff = sm.get_frame_similarity(curr_frame.copy(), last_frame, size=(256, 256), vis=False)
            if diff > thresh:
                frame_file = os.path.join(out_dir, "{}_{:0=6d}.jpg".format(name, count))
                last_frame = curr_frame.copy()
                cv2.imwrite(frame_file, curr_frame)
            if vis:
                text = "TH={},diff={:3.3f}".format(thresh, diff)
                image = image_utils.draw_text(curr_frame, point=(10, 100), color=(0, 255, 0),
                                              text=text, drawType="simple")
                image = image_utils.cv_show_image("image", image, delay=5)
        count += 1
    video_cap.release()
    cv2.destroyAllWindows()


thresh_dict = {"1号视角": 0.5, "2号视角": 0.15, "3号视角": 0.3, "4号视角": 0.3}


def video2frames_demo(root, out, thresh=0.3):
    files = file_utils.get_files_list(root, postfix=["*.avi", "*.mp4", "*.flv"])
    for video_file in tqdm(files):
        # video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/中山视频/开发数据/室外作业视频_0713-0729/20220715室外作业/20220715室外作业01/4号视角.flv"
        print(video_file)
        name = os.path.basename(video_file).split(".")[0]
        if name in thresh_dict:
            thresh = thresh_dict[name]
        video2frames_similarity(video_file, out_dir=out, func=None, interval=100, thresh=thresh, vis=True)


if __name__ == "__main__":
    root = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/中山视频/开发数据/室外作业视频_0713-0729/20220721室外作业/20220721室外作业05"
    out =  "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/中山视频/开发数据/室外作业视频_0713-0729/20220721室外作业-frame"
    video2frames_demo(root, out)
