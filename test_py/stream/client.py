import numpy as np
import cv2, threading, time
from socket import *


def send_img():
    s.sendto(send_data, addr)
    print(f'已发送{img.size}Bytes的数据')
    s.close()
    # input('>>')


# addr = ('192.168.43.106', 8080)
addr = ('127.0.0.1', 8080)          # 127.0.0.1表示本机的IP，相当于我和“自己”的关系
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    s = socket(AF_INET, SOCK_DGRAM)
    th = threading.Thread(target=send_img)
    th.setDaemon(True)

    _, send_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    th.start()
    cv2.putText(img, "client", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('client_frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()