import cv2
import time
import numpy as np
import subprocess
import json


def get_video_size(video):
    """获取视频原始分辨率"""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video]
    result = subprocess.check_output(cmd).decode()
    info = json.loads(result)
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            return stream['width'], stream['height']
    return None, None


class CameraCapture(object):
    def __init__(self, video: str or int = 0, fps=30, size=(1920, 1080), scale=1.0, pad=False):
        """
        查询视频设备分辨率： ffmpeg -f v4l2 -list_formats all -i /dev/video0
        常见的视频分辨率  ： 1920x1080 1280x720 640x480 352x288 320x240 176x144 160x120
        :param video: 视频设备路径或索引（如 0 或 "/dev/video0"）
        :param fps: 视频帧率
        :param size: 视频分辨率 (宽, 高)，(1280,720),(1920,1080)
        :param scale: 视频缩放比例
        :param pad: 是否保持原始视频比例并填充到指定分辨率
        """
        self.fps = fps
        self.stopped = False
        if isinstance(video, int): video = f"/dev/video{video}"
        # self.dsize = size
        # self.ssize = get_video_size(video)
        self.ssize = (320,240)
        self.dsize = self.ssize
        # self.dsize = (640,480)
        # TODO FFmpeg 命令
        # -re: 以原生帧率读取（模拟直播流）
        # -fflags nobuffer: 关键！禁用缓冲区
        # -flags low_delay: 关键！低延迟模式
        # -probesize 32: 减小探测包大小，加快启动
        # -pix_fmt bgr24: 直接输出 BGR 格式，方便 OpenCV/Numpy 使用，避免后续转换
        command = [
            'ffmpeg',
            '-re',  # 按帧率读取
            '-fflags', 'nobuffer',  # 无缓冲
            '-flags', 'low_delay',  # 低延迟
            '-probesize', '32',  # 快速探测
            '-i', video,  # 输入设备
            '-f', 'rawvideo',  # 输出原始视频流
            '-pix_fmt', 'bgr24',  # 像素格式 BGR (OpenCV 格式)
            '-video_size', f"{self.ssize[0]}x{self.ssize[1]}",  # 视频分辨率
            '-r', str(fps),  # 帧率
            '-'  # 输出到 stdout
        ]
        # 启动进程
        print(f"command: {' '.join(command)}")
        self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

    def read(self):
        if self.stopped:
            self.stop()
            return False, None
        # 读取一帧的数据量 (宽 * 高 * 3通道)
        buf = self.pipe.stdout.read(self.dsize[0] * self.dsize[1] * 3)
        if len(buf) != self.dsize[0] * self.dsize[1] * 3:
            self.stop()
            return False, None  # 读取失败或结束
        # TODO 将字节流转换为numpy数组图像(bgr)
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape((self.dsize[1], self.dsize[0], 3))
        return True, bgr

    def release(self):
        self.stop()

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.pipe.terminate()  # 终止进程
        try:
            outs, errs = self.pipe.communicate(timeout=2)
        except subprocess.TimeoutExpired as err:
            # 如果 communicate 也超时（极少见），强制杀死
            self.pipe.kill()
            self.pipe.communicate()  # 再次尝试清理
            print(err)

    def display(self, title="camera", delay=30):
        while True:
            t1 = time.time()
            ret, frame = self.read()
            t2 = time.time()
            if not ret:
                break
            t21 = (t2 - t1) * 1000
            print(f"image shape: {frame.shape},耗时: {t21:.3f}ms")
            cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(title, frame)
            cv2.waitKey(delay)


if __name__ == '__main__':
    fps = 10
    video = 0  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # video = "/home/PKing/Videos/video1.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # video = "/home/PKing/Videos/demo-src.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # cap = CameraCapture(video=video, size=(640, 640), scale=0.5, pad=True, fps=fps)
    cap = CameraCapture(video=video, size=(), scale=1.0, fps=fps)
    cap.display()
