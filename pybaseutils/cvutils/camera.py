import cv2
import time
import numpy as np
import subprocess
import json
import numbers

def get_video_size(video, size=()):
    """
    获取视频设备或文件的分辨率
    :param video: 视频设备路径/索引 或 视频文件路径
    :param size: 要设置的摄像头分辨率 (宽, 高)，仅对摄像头设备有效
    :return: tuple: (width, height) 或 (None, None)
    """
    if isinstance(video, int):
        video = f"/dev/video{video}"
    # 摄像头设备且需要设置分辨率
    if video.startswith('/dev/video') and size:
        cap = cv2.VideoCapture(video)
        try:
            if not cap.isOpened():
                print(f"无法打开摄像头: {video}")
                return None, None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width, height) != size:
                print(f"请求分辨率: {size}，实际分辨率: {width}x{height}")
            else:
                print(f"摄像头分辨率设置为: {width}x{height}")

            return width, height
        except Exception as e:
            print(f"获取摄像头分辨率失败: {e}")
            return None, None
        finally:
            cap.release()

    # 使用 ffprobe 获取分辨率
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        info = json.loads(result)

        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                width, height = stream.get('width'), stream.get('height')
                if width and height:
                    print(f"video={video}, width:{width}, height:{height}")
                    return width, height

        print(f"未找到视频流: {video}")
        return None, None

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"ffprobe 获取分辨率失败: {e}")
        return None, None
    except FileNotFoundError:
        print("错误：未找到 ffprobe，请安装 ffmpeg")
        return None, None
    except Exception as e:
        print(f"获取视频分辨率失败: {e}")
        return None, None


class CameraCapture(object):
    def __init__(self, video: str or int = 0, fps=30, size=(1920, 1080), scale=1.0):
        """
        查询视频设备分辨率： ffmpeg -f v4l2 -list_formats all -i /dev/video0
        常见的视频分辨率  ： 1920x1080 1280x720 640x480 352x288 320x240 176x144 160x120
        :param video: 视频设备路径或索引（如 0 或 "/dev/video0"）
        :param fps: 视频帧率
        :param size: 视频分辨率 (宽, 高)，(1280,720),(1920,1080)
        :param scale: 视频缩放比例
        """
        self.fps = fps
        self.stopped = False
        self.ssize = get_video_size(video, size=size)
        self.dsize = self.ssize
        if isinstance(video, int):  # TODO 摄像头
            video = f"/dev/video{video}"
            self.video_size = f"{self.ssize[0]}x{self.ssize[1]}"  # 视频文件没有这个参数
        else:  # TODO 视频文件
            self.video_size = None
        if size:
            self.dsize = (int(size[0] * scale), int(size[1] * scale))
        else:
            self.dsize = (int(self.dsize[0] * scale), int(self.dsize[1] * scale))
        vf = f'scale={self.dsize[0]}:{self.dsize[1]}'
        # TODO FFmpeg 命令
        # -re: 以原生帧率读取（模拟直播流）
        # -fflags nobuffer: 关键！禁用缓冲区
        # -flags low_delay: 关键！低延迟模式
        # -probesize 32: 减小探测包大小，加快启动
        # -pix_fmt bgr24: 直接输出 BGR 格式，方便 OpenCV/Numpy 使用，避免后续转换
        if self.video_size:  # TODO 如果是摄像头
            command = [
                'ffmpeg',
                '-re',  # 按帧率读取
                '-fflags', 'nobuffer',  # 无缓冲
                '-flags', 'low_delay',  # 低延迟
                '-probesize', '32',  # 快速探测
                '-video_size', f"{self.ssize[0]}x{self.ssize[1]}",  # 视频分辨率
                '-i', video,  # 输入设备
                '-f', 'rawvideo',  # 输出原始视频流
                '-pix_fmt', 'bgr24',  # 像素格式 BGR
                '-vf', vf,  # 分辨率
                '-r', str(fps),  # 帧率
                '-'  # 输出到 stdout
            ]
        else:
            command = [
                'ffmpeg',
                '-re',  # 按帧率读取
                '-fflags', 'nobuffer',  # 无缓冲
                '-flags', 'low_delay',  # 低延迟
                '-probesize', '32',  # 快速探测
                '-i', video,  # 输入设备
                '-f', 'rawvideo',  # 输出原始视频流
                '-pix_fmt', 'bgr24',  # 像素格式 BGR
                '-vf', vf,  # 分辨率
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
    from pybaseutils.cvutils import video_utils

    fps = 10
    # video = video_utils.get_usb_camera()
    video = "/home/PKing/Videos/video1.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # video = "../../data/video/kunkun_cut.mp4"  # Windows 下可能是 0 或 "video=Integrated Webcam"
    # cap = CameraCapture(video=video, size=(), scale=1.0, fps=fps)
    cap = CameraCapture(video=video, size=(1920, 1080), scale=1.0, fps=fps)
    # cap = CameraCapture(video=video, size=(), scale=2,fps=fps)
    cap.display()
