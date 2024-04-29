# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-25 17:42:55
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


def rename_image_file(video_file, image_dir, filename, output, prefix="", vis=True):
    video_cap = image_utils.get_video_capture(video_file)
    width, height, num_frames, fps = image_utils.get_video_info(video_cap)
    file_list = file_utils.read_data(filename, split=None)
    file_utils.create_dir(output)
    for i, file in tqdm(enumerate(file_list)):
        image_file = os.path.join(image_dir, file)
        image_id = os.path.basename(file).split(".")[0]
        count = int(image_id.split("_")[-1])
        # 设置抽帧的位置
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        isSuccess, frame = video_cap.read()
        if isSuccess and os.path.exists(image_file):
            srcfile = os.path.join(output, f"{image_id}_1x.jpg")
            outfile = os.path.join(output, f"{image_id}_6x.jpg")
            image = cv2.imread(image_file)
            cv2.imwrite(outfile, frame)
            file_utils.copy_file(image_file, srcfile)
            diff = np.asarray(image, dtype=np.float32) - np.asarray(frame, dtype=np.float32)
            diff = np.asarray(np.abs(diff), dtype=np.uint8)
            # com = image_utils.image_vstack([image, frame, diff])
            com = image_utils.image_hstack([image, frame])
            # image_utils.cv_show_image("image-frame", com, delay=0)


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/video-compress/JPEGImages"
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/video-compress/camera3.txt"
    # video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/POC/南方电网视频/我的文件/室内/平视-3号位.mp4"
    video_file = "/home/PKing/nasdata/dataset-dmai/AIJE/岗评项目数据/东莞视频/室内/20231027_第二场"
    output = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/video-compress/compress/camera3"

    rename_image_file(video_file, image_dir, filename, output)
