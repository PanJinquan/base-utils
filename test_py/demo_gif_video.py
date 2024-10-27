# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-24 14:03:56
    @Brief  :
"""
from pybaseutils import image_utils, file_utils
from pybaseutils.cvutils import video_utils


def crop_fun(frame):
    """1920 × 1080"""
    h, w = frame.shape[:2]
    frame = image_utils.get_box_crop(frame, box=[60, 5, 1280, 815])
    return frame


def resize_android(frame):
    h, w = frame.shape[:2]
    frame = image_utils.resize_image(frame, size=(None, 580))  # android-video
    return frame


def resize_fun(frame):
    h, w = frame.shape[:2]
    # frame = image_utils.resize_image(frame, size=(224, None))
    frame = image_utils.resize_image(frame, size=(512, None))
    # frame = image_utils.resize_image(frame, size=(960, None))
    # frame = image_utils.resize_image(frame, size=(None, 640))  # android-video
    return frame


def android_gif():
    # video_file = "/media/dm/新加卷/SDK/CSDN/双目测距/demo/image-nouse-wls.mp4"
    video_file = "/home/PKing/nasdata/tmp/tmp/WaterMeter/code/android-demo.mp4"
    video_utils.video2gif(video_file, interval=6, func=resize_android, fps=10, use_pil=False, vis=True)


def python_gif():
    video_file = "/home/PKing/nasdata/tmp/tmp/WaterMeter/code/water-meter-dets.avi"
    video_utils.video2gif(video_file, interval=6, func=resize_fun, fps=8, use_pil=False, vis=True)


def image_gif():
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/方案图/开关检测/images"
    gif_file = image_dir + ".gif"
    frames = file_utils.get_images_list(image_dir)
    image_utils.image_file2gif(frames, size=(416, 416), padding=True, interval=1, gif_file=gif_file, fps=1,
                               use_pil=False)


if __name__ == "__main__":
    image_gif()
    # android_gif()
    # python_gif()
