import cv2
import time
import numpy as np
import subprocess


class CameraCapture(object):
    def __init__(self, video: str or int = 0, size=(1920, 1080), fps=30):
        """
        :param video: 视频设备路径或索引（如 0 或 "/dev/video0"）
        :param size: 视频分辨率 (宽, 高)，(1280,720),(1920,1080)
        :param fps: 视频帧率
        """
        self.size = size
        self.fps = fps
        self.stopped = False
        if isinstance(video, int): video = f"/dev/video{video}"
        # 构建 FFmpeg 命令
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
            '-s', f'{size[0]}x{size[1]}',  # 分辨率
            '-r', str(fps),  # 帧率
            '-'  # 输出到 stdout
        ]
        # 启动进程
        self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)
        print(f"command: {' '.join(command)}")

    def read(self):
        if self.stopped:
            self.stop()
            return False, None
        # 读取一帧的数据量 (宽 * 高 * 3通道)
        buf = self.pipe.stdout.read(self.size[0] * self.size[1] * 3)
        if len(buf) != self.size[0] * self.size[1] * 3:
            self.stop()
            return False, None  # 读取失败或结束
        # TODO 将字节流转换为numpy数组图像(bgr)
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape((self.size[1], self.size[0], 3))
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
    fps = 30
    width = 1920
    height = 1080
    video = 0  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # video = "/home/PKing/Videos/demo-src.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    cap = CameraCapture(video=video, size=(width, height), fps=fps)
    cap.display()
