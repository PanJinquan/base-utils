# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-12-17 15:02:18
    @Brief  :
"""
import cv2
import time
from datetime import datetime, timedelta


def display_hikvision_stream(rtsp_url):
    print(f"正在连接摄像头: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("无法连接到摄像头")
        return

    # 获取视频信息
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("视频流信息:")
    print(f"编码格式: {codec}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps}")

    # 获取起始时间
    start_time = datetime.now()
    start_timestamp = time.time() * 1000  # 转换为毫秒

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # 获取时间戳信息
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time = datetime.now()

                # 计算摄像头实际时间
                # if timestamp_ms == 0:  # 如果摄像头没有返回有效时间戳
                #     elapsed_ms = (time.time() * 1000) - start_timestamp
                #     camera_datetime = start_time + timedelta(milliseconds=elapsed_ms)
                # else:
                camera_datetime = start_time + timedelta(milliseconds=timestamp_ms)

                # 格式化时间字符串
                camera_time_str = camera_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                system_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # 在图像上显示时间戳
                cv2.putText(frame, f"Camera Time: {camera_time_str}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"System Time: {system_time_str}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示视频帧
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow('Camera', frame)

                # 打印时间戳信息到控制台
                print(f"\r摄像头时间: {camera_time_str} | 系统时间: {system_time_str}",
                      end="", flush=True)

                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("\n视频帧获取失败，正在重新连接...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n程序已退出")


if __name__ == "__main__":
    camera_url = "rtsp://admin:C2332416@192.168.2.35/Streaming/Channels/1"
    # display_hikvision_stream(camera_url)
    v = (100, 100) + (100, 100)
    print(v)
