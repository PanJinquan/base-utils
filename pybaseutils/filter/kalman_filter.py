# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-01-19 19:29:47
"""
import cv2
import numpy as np


class KalmanFilter():
    """
    卡尔曼滤波，可以有效解决关键点的抖动问题
    尔曼滤波模型假设k时刻的真实状态是从(k − 1)时刻的状态演化而来，符合下式：
       X(k) = F(k) * X(k-1) + B(k)*U(k) + W（k）
    其中：
       F(k)  是作用在xk−1上的状态变换模型（/矩阵/矢量）。
       B(k)  是作用在控制器向量uk上的输入－控制模型。
       W(k)  是过程噪声，并假定其符合均值为零，协方差矩阵为Qk的多元正态分布
    """

    def __init__(self, stateNum=4, measureNum=2):
        self.kf = cv2.KalmanFilter(stateNum, measureNum)
        # 转移矩阵A
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        # 测量矩阵H
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]],
                                             np.float32)
        # 测量噪声方差矩阵R,取值越小收敛越快
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3

        # 过程(系统)噪声噪声方差矩阵Q
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 1e-6
        # 后验错误估计协方差矩阵P
        self.kf.errorCovPost = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32) * 1
        self.mMeasurement = np.zeros((measureNum,), dtype=np.float32)

    def update(self, point):
        """
        :param point: (2,)
        :return:
        """
        if point[0] > 0 and point[1] > 0:
            mMeasurement = np.array([[point[0]], [point[1]]], dtype=np.float32)
            point = self.kf.correct(mMeasurement)

    def predict(self):
        point = self.kf.predict()
        point = np.asarray([point[0], point[1]]).reshape(-1)
        return point
