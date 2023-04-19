# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : maxmin_distance.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-02-14 18:41:30
"""
import math
import numpy as np
from pybaseutils.cluster import similarity


def maxmin_distance_cluster(data, Theta, minDistanceTH=None, SimilarityType='SIME_Euclidean'):
    '''
    :param data: 输入样本数据,每行一个特征
    :param Theta:阈值，一般设置为0.5，阈值越小聚类中心越多
    :param minDistanceTH: 最小阈值，
            判决条件距离阈值关系：distanceTH=max(maxDistance * Theta,minDistanceTH)
    :param SimilarityType: SIME_Euclidean,SIME_MaxDiff
    :return:样本分类，聚类中心
    '''
    maxDistance = 0
    start = 0  # 初始选一个中心点
    index = start  # 相当于指针指示新中心点的位置
    k = 0  # 中心点计数，也即是类别

    dataNum = len(data)
    distance = np.zeros((dataNum,))
    minDistance = np.zeros((dataNum,))
    classes = np.zeros((dataNum,), dtype=np.int32)
    centerIndex = [index]

    # 初始选择第一个为聚类中心点
    ptrCen = data[0]
    # 寻找第二个聚类中心，即与第一个聚类中心最大距离的样本点
    for i in range(dataNum):
        ptr1 = data[i]
        d = similarity.featureSimilarity(ptr1, ptrCen, SimilarityType)
        distance[i] = d
        classes[i] = k + 1
        if (maxDistance < d):
            maxDistance = d
            index = i  # 与第一个聚类中心距离最大的样本

    minDistance = distance.copy()
    maxVal = maxDistance

    if minDistanceTH is None:
        distanceTH = maxDistance * Theta
        minDistanceTH = 0
    else:
        # distanceTH=max(maxDistance * Theta,minDistanceTH)
        distanceTH = minDistanceTH
    print("maxDistance:{:3.4f},minDistanceTH:{:3.4f},distanceTH:{:3.4f}".format(maxDistance, minDistanceTH, distanceTH))
    while maxVal > distanceTH:
        k = k + 1
        centerIndex += [index]  # 新的聚类中心
        for i in range(dataNum):
            ptr1 = data[i]
            ptrCen = data[centerIndex[k]]
            d = similarity.featureSimilarity(ptr1, ptrCen, SimilarityType)
            distance[i] = d
            # 按照当前最近临方式分类，哪个近就分哪个类别
            if minDistance[i] > distance[i]:
                minDistance[i] = distance[i]
                classes[i] = k + 1
        # 寻找minDistance中的最大距离，若maxVal > (maxDistance * Theta)，则说明存在下一个聚类中心
        index = np.argmax(minDistance)
        maxVal = minDistance[index]
    classes = classes.tolist()
    return classes, centerIndex


if __name__ == '__main__':
    data = [[0, 0], [3, 8], [2, 2], [1, 1], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]]
    Theta = 0.5
    data = np.array(data, dtype=np.float32)
    classes, centerIndex = maxmin_distance_cluster(data, Theta, minDistanceTH=None, SimilarityType="SIME_Euclidean")
    print(classes)
    print(centerIndex)
