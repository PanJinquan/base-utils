# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-09-02 19:04:10
"""
import numpy as np
from sklearn.cluster import KMeans


def sklearn_kmeans(feature, n_clusters, max_iter=300):
    '''
    :param feature:
    :param n_clusters: 聚类的类别个数
    :param max_iter:  聚类最大循环次数
    :return:
    '''
    nums = len(feature)
    input_data = np.array(feature)
    input_data = np.reshape(input_data, newshape=(nums, -1))
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter)  # 分为k类, 并发数4
    model.fit(input_data)  # 开始聚类
    # 简单打印结果
    label = model.labels_
    # r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    # r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    return label


def distEclud(vecA, vecB):
    '''
    计算样本到聚类中心的欧几里得距离
    :param vecA:
    :param vecB:
    :return:
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离


def createRandomCent(dataSet, k):
    '''
    构建聚簇中心，取k个随机质心
    :param dataSet:
    :param k:
    :return:
    '''
    seeds = 2  # 固定种子,只要seed的值一样，后续生成的随机数都一样
    n = np.shape(dataSet)[1]
    centroids = np.zeros(shape=(k, n))  # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        np.random.seed(seeds)
        c = minJ + rangeJ * np.random.rand(k, 1)
        centroids[:, j] = c.reshape(-1)
    return centroids


if __name__ == '__main__':
    data = [[0, 0], [3, 8], [2, 2], [1, 1], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]]
    feature = np.array(data, dtype=np.float32)
    n_clusters = 3
    label = sklearn_kmeans(feature, n_clusters, max_iter=300)
    print(label)
