# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-29 17:38:49
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from datetime import datetime


class RTMPProcessor:
    def __init__(self, url, out_video, out_frame=None, extract_fps=1):
        """
        初始化RTMP处理器
        :param url: RTMP流地址
        :param out_video: 输出/保存MP4文件路径
        :param out_frame: 输出/保存抽帧jpg图片根目录
        :param extract_fps: 抽帧频率
        """
        self.url = url
        if out_video: os.makedirs(os.path.dirname(out_video), exist_ok=True)
        if out_frame: os.makedirs(out_frame, exist_ok=True)
        self.out_video = out_video if out_video else None
        self.out_frame = os.path.join(out_frame, 'frame_%04d.jpg') if out_frame else None
        self.count = 0  # 处理统计
        self.log = print
        self.extract_fps = extract_fps
        self.t1 = datetime.now()
        self.t2 = datetime.now()

    def setup_ffmpeg_pipes(self):
        """设置FFmpeg输入输出管道"""
        # FFmpeg命令从RTMP读取
        self.stream_inp = subprocess.Popen(['ffmpeg',
                                            '-i', self.url,
                                            '-loglevel', 'quiet',  # 减少控制台输出
                                            '-f', 'image2pipe',  # 指定输出格式为图像管道
                                            '-pix_fmt', 'bgr24',  # OpenCV兼容的像素格式
                                            '-vcodec', 'rawvideo',
                                            '-'  # 表示输出到标准输出(stdout)而不是文件
                                            ],
                                           stdout=subprocess.PIPE,  # 将输出通过管道传递给Python程序
                                           bufsize=10 ** 8)

        # 获取视频信息
        probe_cmd = ['ffprobe',
                     '-v', 'error',
                     '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,pix_fmt,r_frame_rate',
                     '-of', 'csv=p=0',
                     self.url
                     ]
        info = subprocess.check_output(probe_cmd).decode('utf-8').strip().split(',')
        if len(info) < 4: raise ValueError("无法获取视频流信息")
        self.width = int(info[0])
        self.height = int(info[1])
        self.pix_fmt = info[2]
        self.fps = eval(info[3])  # 解析帧率(如30/1)
        self.log(f"视频信息: {self.width}x{self.height} {self.pix_fmt} {self.fps}FPS")
        # FFmpeg命令写入MP4
        self.stream_out = subprocess.Popen(['ffmpeg',
                                            '-y',  # 覆盖输出文件
                                            '-f', 'rawvideo',
                                            '-vcodec', 'rawvideo',
                                            '-s', f'{self.width}x{self.height}',
                                            '-pix_fmt', 'bgr24',
                                            '-r', str(self.fps),  # 保持原始帧率
                                            '-i', '-',  # 从管道输入
                                            '-i', self.url,  # 原始音频流
                                            '-map', '0:v',  # 处理后的视频
                                            '-map', '1:a',  # 原始音频
                                            '-c:v', 'libx264',  # H.264编码
                                            '-preset', 'fast',
                                            '-crf', '23',
                                            '-c:a', 'aac',  # AAC音频编码
                                            '-b:a', '128k',
                                            '-shortest',  # 以最短的流结束
                                            self.out_video]
                                           + list(['-vf', f'fps={self.extract_fps}',
                                                   '-q:v', '2',  # 控制输出质量(2-31，越低越好)
                                                   self.out_frame] if self.out_frame and self.extract_fps > 0 else []),
                                           stdin=subprocess.PIPE)

    def task(self, frame):
        """应用高斯模糊效果"""
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def process_frames(self):
        """处理视频帧"""
        try:
            pbar = tqdm(bar_format="{desc}")  # 初始化总进度
            while True:
                # 从FFmpeg输入管道读取原始帧
                buffer = self.stream_inp.stdout.read(self.width * self.height * 3)
                if not buffer:
                    self.log("视频流结束")
                    break
                # 转换为numpy数组
                frame = np.frombuffer(buffer, dtype='uint8').reshape((self.height, self.width, 3))
                frame = self.task(frame)
                # 写入FFmpeg输出管道
                self.stream_out.stdin.write(frame.tobytes())
                self.count += 1
                self.t2 = datetime.now()
                fps = self.count / (self.t2 - self.t1).total_seconds()
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = f"{time} Process {self.count:5d}帧, fps={fps:.2f}"
                pbar.set_description_str(msg)  # 设置描述
        except Exception as e:
            self.log(f"处理过程中发生错误: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'stream_inp'):
            self.stream_inp.stdout.close()
            self.stream_inp.terminate()

        if hasattr(self, 'stream_out'):
            self.stream_out.stdin.close()
            self.stream_out.terminate()

        elapsed = (datetime.now() - self.t1).total_seconds()
        self.log(f"处理完成,总共处理 {self.count} 帧")
        self.log(f"总耗时: {elapsed:.2f} 秒")
        self.log(f"平均FPS: {self.count / elapsed:.2f}")
        self.log(f"输出文件已保存到: {self.out_video}")

    def run(self):
        """启动处理流程"""
        try:
            self.setup_ffmpeg_pipes()
            self.process_frames()
        except KeyboardInterrupt:
            self.log("用户中断处理")
            self.cleanup()
        except Exception as e:
            self.log(f"初始化失败: {str(e)}")
            self.cleanup()
            raise


if __name__ == "__main__":
    # 配置参数
    url = "/media/PKing/新加卷/SDK/base-utils/data/video/test-video.mp4"  # 替换为实际的RTMP流地址
    url = "/home/PKing/Videos/aije-video/主视.mp4"  # 替换为实际的RTMP流地址
    out_video = "/home/PKing/Videos/output.mp4"  # 输出文件名
    out_frame = "/home/PKing/Videos/output"  # 输出文件名
    BLUR_LEVEL = 15  # 模糊程度(奇数)

    # 创建并运行处理器
    processor = RTMPProcessor(url, out_video, out_frame)
    processor.run()
