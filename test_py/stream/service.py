import numpy as np
import cv2, threading, time
from socket import *


def send_img():
    t0 = time.time()
    s.sendto(send_data, addr)
    s.close()
    t1 = time.time()
    elapsed = (t1 - t0) * 1000
    print('已发送{}Bytes的数据,elapsed:{:3.3f}ms'.format(img.size, elapsed))


addr = ('192.168.2.219', 8080)
# addr = ('127.0.0.1', 8080)          # 127.0.0.1表示本机的IP，相当于我和“自己”的关系
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
