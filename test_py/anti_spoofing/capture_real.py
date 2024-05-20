# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   :
# 安装方法 ：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python numpy
# --------------------------------------------------------
"""

import sys
import os
import argparse
import utils


def get_parser():
    name = "0001"  # 人员ID，即可
    rgb_video = "/home/PKing/nasdata/FaceDataset/anti-spoofing/demo-video/真人脸视频(副本)/彩色.avi"
    inf_video = "/home/PKing/nasdata/FaceDataset/anti-spoofing/demo-video/真人脸视频(副本)/红外.avi"
    out_root = "/home/PKing/nasdata/FaceDataset/anti-spoofing/demo-video/StereoCamera"
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument("--name", type=str, help="每人唯一ID，如0001，0002，0003等", default=name)
    parser.add_argument("--prefix", type=str, help="数据类型real,paper(纸),monitor(显示屏),mask(面具)", default="real")
    parser.add_argument('--rgb_video', type=str, default=rgb_video, help='RGB摄像头USB的ID,一般是0,1,2')
    parser.add_argument('--inf_video', type=str, default=inf_video, help='红外摄像头USB的ID,一般是0,1,2')
    parser.add_argument('--out_root', type=str, default=out_root, help='保存视频路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    print(args)
    sd = utils.StereoCamera(args.rgb_video, args.inf_video)
    sd.capture(args.out_root, name=args.name, prefix=args.prefix)
