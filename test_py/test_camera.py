import numpy as np
import cv2
import time
from pybaseutils.cvutils.camera import CameraCapture
from pybaseutils.cvutils.video_utils import get_usb_camera


def display(cap: CameraCapture):
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        t2 = time.time()
        if not ret:
            break
        t21 = (t2 - t1) * 1000
        print(f"image shape: {frame.shape},耗时: {t21:.3f}ms")
        cv2.imshow('FFmpeg Low Latency', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    fps = 15
    width = 1920
    height = 1080
    # video = 0  # Windows 下可能是 0 或 "video=Integrated Webcam"
    video = "/home/PKing/Videos/demo-src.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    video = get_usb_camera()
    cap = CameraCapture(video=video, size=(width, height), fps=fps)
    display(cap)
