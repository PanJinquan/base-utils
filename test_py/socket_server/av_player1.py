import cv2  # 导入OpenCV库，用于显示视频帧
import av  # 导入av库，用于读取和解码音视频文件
import pyaudio  # 导入pyaudio库，用于播放音频
import numpy as np  # 导入numpy库，用于处理数组


def play_video(video_path):  # 定义一个函数play_video，接受视频文件路径作为参数
    container = av.open(video_path)  # 打开指定路径的视频文件

    # 初始化音频流
    audio_stream = next(s for s in container.streams if s.type == 'audio')  # 查找并获取第一个音频流
    audio_format = pyaudio.paInt16  # 设置音频格式为16位整数
    channels = audio_stream.channels  # 获取音频通道数
    rate = audio_stream.rate  # 获取音频采样率

    p = pyaudio.PyAudio()  # 创建PyAudio对象
    audio_buffer = bytearray()  # 初始化音频缓冲区为bytearray以便于拼接

    def callback(in_data, frame_count, time_info, status):  # 定义回调函数，用于向pyaudio提供音频数据
        print(1)
        nonlocal audio_buffer  # 声明使用非局部变量audio_buffer
        data = bytes(audio_buffer[:frame_count * channels * (audio_format // 8)])  # 从音频缓冲区中取出所需的数据
        audio_buffer = audio_buffer[frame_count * channels * (audio_format // 8):]  # 删除已使用的音频数据
        return (data, pyaudio.paContinue)  # 返回音频数据和继续标志

    audio_stream_pyaudio = p.open(format=audio_format,  # 打开音频流
                                  channels=channels,
                                  rate=rate,
                                  output=True,
                                  stream_callback=callback)

    # 初始化视频显示
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # 创建一个名为'Video'的窗口，大小可调整

    for packet in container.demux():  # 遍历容器中的所有数据包
        for frame in packet.decode():  # 解码数据包中的每一帧
            if isinstance(frame, av.AudioFrame):  # 如果是音频帧
                audio_buffer.extend(frame.to_ndarray().tobytes())  # 将音频帧转换为字节并添加到缓冲区
            elif isinstance(frame, av.VideoFrame):  # 如果是视频帧
                img = frame.to_image()  # 将视频帧转换为图像
                img_np = np.array(img)  # 将图像转换为numpy数组
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 将颜色空间从RGB转换为BGR
                # cv2.imshow('Video', img_np)  # 在窗口中显示图像

                # if cv2.waitKey(1) & 0xFF == ord('q'):  # 检查是否按下'q'键
                #     break  # 如果按下'q'键，则退出循环

    audio_stream_pyaudio.stop_stream()  # 停止音频流
    audio_stream_pyaudio.close()  # 关闭音频流
    p.terminate()  # 终止PyAudio对象
    # cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


if __name__ == "__main__":
    # video_path = 'example.mp4'  # 替换为你自己的视频文件路径
    url = "/home/PKing/Videos/dde-introduction.mp4"
    play_video(url)  # 调用play_video函数播放视频



