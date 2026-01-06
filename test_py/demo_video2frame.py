# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils
from pybaseutils.cvutils import video_utils


def video2frames_demo(root, out, vis=False):
    out = file_utils.create_dir(out)
    files = file_utils.get_files_lists(root, postfix=file_utils.VIDEO_POSTFIX)
    print("save:{}".format(out))
    for video_file in files:
        print(video_file)
        # video_utils.video2frames_similarity(video_file, out_dir=out, func=None, interval=20, thresh=0.3, vis=True)
        video_utils.video2frames(video_file, out_dir=out, task=None, interval=20, vis=vis, prefix="")


if __name__ == "__main__":
    root = "/home/PKing/edudata/dataset/AIJE/【TOP】技能人才系统_数据集管理/06-联通岗评/02-训练数据/联通入户布线_156_20251219_180523-胶枪打胶专题数据/20251219_180523_主视.mp4"
    out = os.path.join(os.path.dirname(root), "frames")
    # out = os.path.join(root, "frames")
    video2frames_demo(root, out)
