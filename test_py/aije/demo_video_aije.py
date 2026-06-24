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
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.cvutils import video_utils, frame_record

# thresh_dict = {"1号视角": 0.5, "2号视角": 0.15, "3号视角": 0.3, "4号视角": 0.3}
# thresh_dict = {"一号位": 0.58, "平视": 0.35, "俯视": 0.35, "右视": 0.25, "左视": 0.3}
# thresh_dict = {"主视": 0.50, "平视": 0.30, "俯视": 0.30, "右视": 0.35, "左视": 0.35,"全景": 0.30}
thresh_dict = {"主视": 0.2, "1号": 0.2, "2号": 0.2, "3号": 0.2, "4号": 0.2, "5号": 0.2, "6号": 0.2}


def video2frames_demo(root, out="", prefix=""):
    """S
 SS   :param root:
    :param out:
    :param prefix:
    :return:
    """
    files = file_utils.get_files_lists(root, postfix=["*.avi", "*.mp4", "*.flv"])
    for video_file in tqdm(files):
        print(video_file)
        out_dir = out if out else video_file.split(".")[0]
        name = os.path.basename(video_file).split(".")[0]
        thresh = thresh_dict.get(name, 0.4)
        # prefix_ = os.path.basename(os.path.dirname(video_file)).replace("-", "_")
        # if prefix: prefix_ = "{}_{}".format(prefix, prefix_)
        video_utils.video2frames_similarity(video_file, out_dir=out_dir, func=None, interval=20, thresh=thresh,
                                            prefix=prefix, vis=True)


if __name__ == "__main__":
    root = "/media/PKing/dev1/project/smart-dinner/dataset2/20251215"
    video2frames_demo(root, prefix="")
