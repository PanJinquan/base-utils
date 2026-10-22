# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-02-18 16:10:49
# @Brief  :
# --------------------------------------------------------
"""
import numpy as np
import cv2
import pose_filter


class FilterDemo():
    def __init__(self):
        self.filter = pose_filter.PoseFilter()
        self.curr_mes = np.array([-1, -1])
        self.curr_pre = np.array([-1, -1])

        self.last_mes = np.array([-1, -1])
        self.last_pre = np.array([-1, -1])
        self.frame = np.zeros((800, 800, 3), np.uint8)

    def mouseEvent(self, event, x, y, s, p):
        self.curr_mes = np.array([x, y], dtype=np.float32)

    def task(self):
        self.curr_pre = np.array(self.curr_mes)
        self.curr_pre = self.filter.update(self.curr_pre.reshape(1,2))
        self.curr_pre = self.curr_pre[0,:]
        curr_mes = (int(self.curr_mes[0]), int(self.curr_mes[1]))
        last_mes = (int(self.last_mes[0]), int(self.last_mes[1]))
        curr_pre = (int(self.curr_pre[0]), int(self.curr_pre[1]))
        last_pre = (int(self.last_pre[0]), int(self.last_pre[1]))
        # 绘制测量值轨迹（绿色）
        cv2.line(self.frame, last_mes, curr_mes, (0, 255, 0))
        # 绘制预测值轨迹（红色）
        cv2.line(self.frame, last_pre, curr_pre, (0, 0, 255))
        print("last_pre:{},curr_pre:{}".format(last_pre, curr_pre))
        self.last_mes = self.curr_mes.copy()
        self.last_pre = self.curr_pre.copy()

    def demo(self):
        cv2.namedWindow("Kalman")
        cv2.setMouseCallback("Kalman", self.mouseEvent)
        while (True):
            # 绘制预测值轨迹（蓝色）
            self.task()
            cv2.imshow('Kalman', self.frame)
            key = cv2.waitKey(100)
            if key == 27:  # ESC对应的ASCII码是27
                break
            elif key == ord("c") or key == ord("C"):
                self.frame = np.zeros((800, 800, 3), np.uint8)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(__name__)
    fd = FilterDemo()
    fd.demo()
