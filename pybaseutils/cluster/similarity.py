# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-09-02 17:04:52
"""

import cv2
import numpy as np
import math


def featureSimilarity(mat1, mat2, SimilarityType):
    '''
    计算相似程度
    :param mat1:type;list
    :param mat2:type:list
    :param SimilarityType:SIMI_CompareHist,SIMI_Cosine,SIME_Euclidean
    :return:
    '''
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    dest = 0
    if SimilarityType == "SIMI_CompareHist":
        dest = calHistSimilarity(mat1, mat2)
    elif SimilarityType == "SIMI_Cosine":
        dest = calCosineSimilarity(mat1, mat2, False)
    elif SimilarityType == "SIME_Euclidean":
        dest = calEuclideanDistance(mat1, mat2)
    elif SimilarityType == "SIME_MaxDiff":
        dest = calMaxDiffDistance(mat1, mat2, Theta=0.5)
    return dest


def calMaxDiffDistance(data1, data2, Theta=0.5):
    '''
    :param data1:
    :param data2:
    :param Theta:阈值0-1
    :return:
    '''
    # TH_diff=np.where(diff>TH,diff,0)
    diff = abs(data1 - data2)
    maxValue = np.max(diff)
    minValue = np.min(diff)
    TH = Theta * (maxValue - minValue)
    sum_s = 0
    for d in diff:
        temp_max = np.max(d)
        if temp_max > TH:
            temp_d = np.square(d)
            sum_s += np.sum(temp_d)
    dest = np.sqrt(sum_s)
    return dest


def calEuclideanDistance(data1, data2):
    '''
    计算欧式距离
    :param data1:
    :param data2:
    :return:
    '''
    # distance = np.sum(np.power(data1-data2, 2), axis=1)
    d = np.square((data1 - data2))
    distance = np.sqrt(np.sum(d))
    # distance = np.sum(d)
    return distance


def calCosineSimilarity(mat1, mat2, bNormalization=False):
    '''
    计算余弦相似性
    :param mat1:
    :param mat2:
    :param bNormalization:
    :return:
    '''
    AB = np.dot(mat1, mat2)
    A1 = np.linalg.norm(mat1)
    B1 = np.linalg.norm(mat2)
    dest = AB / (A1 * B1)
    # d = (Omax - Omin) * (x - Imin) / (Imax - Imin) + Omin;
    if (bNormalization):  # 归一化到0 - 1
        dest = (dest + 1) / 2
    return dest


def calHistSimilarity(mat1, mat2):
    '''
    计算HIST相似性
    :param mat1:
    :param mat2:
    :return:
    '''
    dSimilarity = cv2.compareHist(mat1, mat2)
    if dSimilarity < 0.0:
        dSimilarity = 0.0
    return dSimilarity


if __name__ == "__main__":
    data1 = [[0, 1, 2, 3, 4],
             [5, 6, 7, 8, 9],
             [10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]]

    data2 = [[0, 1, 2, 10, 5],
             [5, 6, 8, 8, 9],
             [11, 11, 20, 13, 13],
             [14, 16, 15, 18, 22]]
    data1 = np.array([1, 2, 3, 4])
    data2 = np.array([1, 1, 2, 2])
    calEuclideanDistance(data1, data2)
