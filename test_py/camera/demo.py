import os
import cv2
import time
import requests
import xmltodict
from datetime import datetime
from requests.auth import HTTPDigestAuth


class CameraDemo():
    def __init__(self, ip, username, password):
        """
        按'q'键退出
        :param ip: IP摄像头IP地址
        :param username: 用户
        :param password: 密码
        """
        self.ip = ip
        self.username = username
        self.password = password

    def get_camera_time(self):
        """获取海康摄像头的时间
        urls = [
                f'http://{ip}/ISAPI/System/time',
                f'http://{ip}/ISAPI/System/status',
                f'http://{ip}/ISAPI/System/deviceInfo'
            ]
        """
        t = "unknown camera time"
        try:
            # 尝试不同的 ISAPI 接口
            url = f'http://{self.ip}/ISAPI/System/time'
            response = requests.get(url, auth=HTTPDigestAuth(self.username, self.password), verify=False, timeout=5)
            content = response.content
            content = xmltodict.parse(content)  # 将XML格式json格式
            if response.status_code == 200:
                t = content['Time']['localTime']
                t = t[:-len('+08:00')]  # 去除+08:00字符
            else:
                print(f"code: {response.status_code},content={response.text}")
        except Exception as e:
            print(f"Error: {e}")
        return t

    def get_camera_info(self, cap):
        """
        获取视频信息
        :param cap:
        :return:
        """
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频流信息, 编码格式:{codec}, 分辨率:{width}x{height}, 帧率:{fps}")
        return width, height, fps

    def draw_image_info(self, frame, info: str, point, color=(0, 255, 0), fontScale=1.2, thickness=2):
        cv2.putText(frame, info, point, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
        return frame

    @staticmethod
    def get_video_writer(video_file, width, height, fps):
        """
        获得视频存储对象
        :param video_file: 输出保存视频的文件路径
        :param width:   图像分辨率width
        :param height:  图像分辨率height
        :param fps:  设置视频播放帧率
        :return:
        """
        if not os.path.exists(os.path.dirname(video_file)):
            os.makedirs(os.path.dirname(video_file))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frameSize = (int(width), int(height))
        video_writer = cv2.VideoWriter(video_file, fourcc, int(fps), frameSize)
        print("save video:width:{},height:{},fps:{}".format(width, height, fps))
        return video_writer

    def capture(self, title="camera", save_video="./camera.avi"):
        format = '%Y-%m-%dT%H:%M:%S'  # 海康摄像头默认格式
        video_url = f"rtsp://{self.username}:{self.password}@{self.ip}/Streaming/Channels/1"
        print(f"正在连接摄像头: {video_url}")
        # 创建视频流连接
        cap = cv2.VideoCapture(video_url)
        assert cap.isOpened(), f"无法连接到摄像头:{video_url}"
        W, H, Fps = self.get_camera_info(cap)
        video_writer = self.get_video_writer(save_video, width=W, height=H, fps=Fps)
        # TODO 获得启动摄像头的时间
        cam_start_time = self.get_camera_time()
        cam_start_timestamp = time.mktime(time.strptime(cam_start_time, format))
        reconnection = 0
        count = 0
        point = (W - 750, 150)
        while True:
            ret, frame = cap.read()
            count += 1
            if ret:
                # TODO 获取当前系统时间
                sys_timestamp = time.time()
                sys_time = datetime.fromtimestamp(sys_timestamp).strftime(format)
                # TODO 通过接口获得摄像头设备时间
                cam_time = self.get_camera_time()
                cam_timestamp = time.mktime(time.strptime(cam_time, format))
                # TODO 通过视频解码估计视频流生成时间
                time.sleep(0.1)  # TODO 模拟网络延时
                vdo_offset = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 获得视频偏移量
                vdo_timestamp = cam_start_timestamp + vdo_offset  # 设备启动时间+视频偏移量得到当前帧的到达时间
                vdo_time = datetime.fromtimestamp(vdo_timestamp).strftime(format)
                # 在图像上显示时间戳
                sys_info = f"System Time: {sys_time}"
                cam_info = f"Camera Time: {cam_time}"
                vdo_info = f"Video  Time: {vdo_time}"
                # 打印时间戳信息到控制台
                print(f"\r{cam_info} | {sys_info}| {vdo_info}", end="", flush=True)
                frame = self.draw_image_info(frame, info=vdo_info, point=(point[0], point[1]))
                frame = self.draw_image_info(frame, info=cam_info, point=(point[0], point[1] + 50))
                frame = self.draw_image_info(frame, info=sys_info, point=(point[0], point[1] + 100))
                # 显示视频帧
                cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
                cv2.imshow(title, frame)
                # 保存视频
                video_writer.write(frame)
                # 按'q'键退出
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                cap.release()
                reconnection += 1
                print(f"视频帧获取失败，尝试第{reconnection}次重新连接...")
                time.sleep(10)
                cap = cv2.VideoCapture(video_url)
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print("\n程序已退出")


if __name__ == "__main__":
    # 摄像头信息，按'q'键退出程序
    ip = "192.168.2.35"
    username = "admin"
    password = "C2332416"
    c = CameraDemo(ip, username, password)
    c.capture()
