# -*-coding: utf-8 -*-
"""
    @File   : numpy_tools.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-10-23 12:01:36
"""
import numpy as np
import math
from sklearn import metrics, preprocessing
import heapq


def feature_norm(x, axis=-1):
    """
    特征归一化,会使得特征L2之和为1, 即||y||=L2(y,axis=-1)
    y = x / x.norm(dim=-1, keepdim=True)               # torch
    y = x / np.linalg.norm(x, axis=-1, keepdims=True)  # numpy
    :param x: 输入二维矩阵(N,embedding-size)，每行是一个样本，样本特征维度为embedding-size
    :param axis:
    :return:
    """
    y = x / np.linalg.norm(x, axis=axis, keepdims=True)
    return y


def torch_norm(x, axis=-1):
    """
    特征归一化:
    y = x / x.norm(dim=-1, keepdim=True)               # torch
    y = x / np.linalg.norm(x, axis=-1, keepdims=True)  # numpy
    :param x: 输入二维矩阵(N,embedding-size)，每行是一个样本，样本特征维度为embedding-size
    :param axis:
    :return:
    """
    import torch
    x = torch.from_numpy(x)
    y = x / x.norm(dim=-1, keepdim=True)  # torch
    return y


def feature_similarity(inputs, target):
    """
    计算特征相似性
    similarity = inputs @ target.T                 # torch
    similarity = np.sum(inputs * target, axis=-1)  # numpy
    :param inputs: 输入待匹配的特点，shape=(n,D),其中n表示样本个数，D表示特征维度
    :param target: 目标匹配数据库,shape=(N,D),其中N表示数据库样本个数，D表示特征维度
    :return: similarity
    """
    assert inputs.shape[1] == target.shape[1]  # 特征维度必须一致
    inputs_ = np.expand_dims(inputs, axis=1)  # (b,512)->(b,1,512)
    target_ = np.expand_dims(target, axis=0)  # (n, 512)->(1, n, 512)
    similarity = np.sum(inputs_ * target_, axis=-1)  # (b,1,512) * (1, n, 512)
    return similarity


def feature_matching(inputs, target, use_max=True):
    """
    按照相似程度进行特征匹配，输入特征数据必须进行特征归一化:
    y = x / x.norm(dim=-1, keepdim=True)               # torch
    y = x / np.linalg.norm(x, axis=-1, keepdims=True)  # numpy
    :param inputs: 输入待匹配的特点，shape=(n,D),其中n表示样本个数，D表示特征维度
    :param target: 目标匹配数据库,shape=(N,D),其中N表示数据库样本个数，D表示特征维度
    :param use_max:按照相似程度进行匹配
    :return: index： 与target匹配的索引
             score：与target匹配的最小L2距离(欧式距离=np.sqrt(L2)=np.sqrt(distance))
    """
    similarity = feature_similarity(inputs, target)
    if use_max:
        index = np.argmax(similarity, axis=1)
        score = np.max(similarity, axis=1)
    else:
        index = np.argmin(similarity, axis=1)
        score = np.min(similarity, axis=1)
    return index, score


def get_nearest_point(points, center, axis=1, use_max=False):
    """
    求离center最近/最远的点
    :param points: shape=(n,D),其中n表示样本个数，D表示特征维度
    :param center: (x,y)
    :param axis:
    :param use_max: False:按照最小距离进行匹配,True:按照最大距离进行匹配
    :return: index： 最近点index
             distance：最近点L2距离
    """
    if not isinstance(points, np.ndarray): points = np.asarray(points)
    # l2 = np.sum(np.square(x), axis=axis)
    l2dist = np.sum(np.square(points - center), axis=axis)  # l2距离，L2开根号就是欧式距离
    if use_max:
        index = np.argmax(l2dist)
        distance = l2dist[index]
    else:
        index = np.argmin(l2dist)
        distance = l2dist[index]
    return index, distance


def get_nearest_point_minmax(points, center, axis=1, use_max=False, minmax=-1):
    """
    求离center最近/最远的点
    :param points: shape=(n,D),其中n表示样本个数，D表示特征维度
    :param center: (x,y)
    :param axis:
    :param use_max: False:按照最小距离进行匹配,True:按照最大距离进行匹配
    :param minmax:  use_max=False,表示查找不低于阈值minmax的最小距离；
                    use_max=True, 表示查找不高于阈值minmax的最大距离；
                    当minmax=-1则，该功能无效
    :return: index： 最近点index
             distance：最近点L2距离
    """
    if not isinstance(points, np.ndarray): points = np.asarray(points)
    l2dist = np.sum(np.square(points - center), axis=axis)  # l2距离，L2开根号就是欧式距离
    num = len(l2dist)
    if use_max:
        indexes = np.argsort(-l2dist)  # 从大到小排列
        index = 0
        distance = l2dist[indexes[index]]
        while minmax > 0 and index < num and distance > minmax:
            index += 1
            distance = l2dist[indexes[index]]
    else:
        indexes = np.argsort(l2dist)  # 从小到大排列
        index = 0
        distance = l2dist[indexes[index]]
        while minmax > 0 and index < num and distance < minmax:
            index += 1
            distance = l2dist[indexes[index]]
    return indexes[index], distance


def get_nearest_point_sort(points, center, axis=1, use_max=False, minmax=-1):
    """
    求离center最近/最远的点
    :param points: shape=(n,D),其中n表示样本个数，D表示特征维度
    :param center: (x,y)
    :param axis:
    :param use_max: False:按照最小距离进行匹配,True:按照最大距离进行匹配
    :param minmax:  use_max=False,表示查找不低于阈值minmax的最小距离；
                    use_max=True, 表示查找不高于阈值minmax的最大距离；
                    当minmax=-1则，该功能无效
    :return: index： 最近点index
             distance：最近点L2距离
    """
    if not isinstance(points, np.ndarray): points = np.asarray(points)
    l2dist = np.sum(np.square(points - center), axis=axis)  # l2距离，L2开根号就是欧式距离
    if use_max:
        indexes = np.argsort(-l2dist)  # 从大到小排列
        distance = l2dist[indexes]
    else:
        indexes = np.argsort(l2dist)  # 从小到大排列
        distance = l2dist[indexes]
    return indexes, distance


def matching_data_vecror(data, vector):
    '''
    从data中匹配vector向量，查找出现vector的index,如：
    data = [[1., 0., 0.],[0., 0., 0.],[2., 0., 0.],
            [0., 0., 0.],[0., 3., 0.],[0., 0., 4.]]
    # 查找data中出现[0, 0, 0]的index
    data = np.asarray(data)
    vector=[0, 0, 0]
    index =matching_data_vecror(data,vector)
    print(index)
    >>[False  True False  True False False]
    # 实现去除data数组中元素为[0, 0, 0]的行向量
    pair_issame_1 = data[~index, :]  # 筛选数组
    :param data:
    :param vector:
    :return:
    '''
    # index = (data[:, 0] == 0) & (data[:, 1] == 0) & (data[:, 2] == 0)
    row_nums = len(data)
    clo_nums = len(vector)
    index = np.asarray([True] * row_nums)
    for i in range(clo_nums):
        index = index & (data[:, i] == vector[i])
    return index


def set_mat_vecror(data, index, vector):
    '''
    实现将data指定index位置的数据设置为vector
    # 实现将大于阈值分数的point，设置为vector = [10, 10]
    point = [[0., 0.], [1., 1.], [2., 2.],
             [3., 3.], [4., 4.], [5., 5.]]
    point = np.asarray(point) # 每个数据点
    score = np.array([0.7, 0.2, 0.3, 0.4, 0.5, 0.6])# 每个数据点的分数
    score_th=0.5
    index = np.where(score > score_th) # 获得大于阈值分数的所有下标
    vector = [10, 10]                  # 将大于阈值的数据设置为vector
    out = set_mat_vecror(point, index, vector)
    :param data:
    :param index:
    :param vector:
    :return:
    '''
    data[index, :] = vector
    return data


def find_max_shape_data(list_data):
    max_shape_data = np.asarray([])
    for data in list_data:
        if len(max_shape_data) < len(data):
            max_shape_data = data
    return max_shape_data


def data_alignment(data):
    '''
    row_stack()函数扩展行，column_stack()函数扩展列
    :param list_data:
    :param align:
    :param extend:
    :return:
    '''
    max_shape_data = find_max_shape_data(data)
    for i in range(len(data)):
        maxdata = np.zeros(shape=max_shape_data.shape, dtype=max_shape_data.dtype) - 1
        shape = data[i].shape
        if len(shape) == 1:
            maxdata[0:shape[0]] = data[i]
        else:
            maxdata[0:shape[0], 0:shape[1]] = data[i]
        data[i] = maxdata
    # data = np.asarray(data)
    return data


def get_batch(image_list, batch_size):
    '''
    batch size data
    :param image_list:
    :param batch_size:
    :return:
    '''
    sample_num = len(image_list)
    batch_num = math.ceil(sample_num / batch_size)

    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_num)
        batch_image = image_list[start:end]
        print("batch_image:{}".format(batch_image))


def gen_range(shape=None, start=None, *args, **kwargs):
    '''
    create range data->
    gen_range(shape=(10, 10), start=0, stop=100)
    :param shape:
    :param start:
    :param args:
    :param kwargs:
    :return:
    '''
    data = np.arange(start, *args, **kwargs)
    if shape:
        data = np.reshape(data, newshape=shape)
    return data


def mat2d_data(data, indexes):
    '''
    get element by indexes
    data=numpy_tools.gen_range((3,3),0,9)
    data=np.mat(data)
    indexes=np.asarray([[1,1],[2,1],[2,2]])
    out=mat2d_data(data, indexes)
    print(data)
    print(out)
    :param data:
    :param indexes:
    :return:
    '''
    out = data[indexes[:, 0], indexes[:, 1]]
    return out


def count_sort_list(list_data: list, reverse=True):
    '''
    给定一个非空正整数的数组，按照数组内数字重复出现次数，从高到低排序
    :param list_data:
    :param reverse: True-降序,Fasle-升序
    :return:
    '''
    d = {}
    list_sorted = []
    for i in list_data:
        d[i] = list_data.count(i)
    # 根据字典值的降序排序
    d_sorted = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    # 输出排序后的数组
    for x in d_sorted:
        for number in range(0, x[1]):
            list_sorted.append(x[0])
    return list_sorted


def remove_list_data(list_data, flag=["", -1]):
    '''
    删除list所有符合条件元素
    :param list_data:
    :param flag:
    :return:
    '''
    for f in flag:
        while f in list_data:
            list_data.remove(f)
    return list_data


def label_alignment(data_list):
    mat = np.asarray(data_list).T
    label_list = []
    for data in mat:
        out = count_sort_list(data.tolist(), reverse=True)
        out = remove_list_data(out, flag=["", -1])
        if out:
            label = out[0]
        else:
            label = -1
        label_list.append(label)
    return label_list


def __print(data, info=""):
    print("-------------------------------------")
    print(info)
    for index in range(len(data)):
        print("{}".format(data[index]))


def euclidean_distance(p1, p2, axis=1):
    """
    计算欧氏距离
    point1 = [[3, 4], [4, 3], [4, 3]]
    center = [[0, 0]]
    point1 = np.asarray(point1)
    center = np.asarray(center)

    L2-norm=np.sqrt(np.sum(np.square(x), axis=axis))
    L1-norm=np.sum(np.abs(x), axis=axis))
    # 下面三个计算是等价的
    d1 = np.sqrt(numpy_tools.l2(point1 - center, axis=1)) # L2开根号就是欧式距离
    d2 = numpy_utils.norm(point1 - center, p=2, axis=1)
    d3 = numpy_utils.euclidean_distance(point1, center, axis=1)
    """
    d = np.sqrt(np.sum(np.square(p1 - p2), axis=axis))
    return d


def norm(x, p=1, axis=0):
    """
    L-p范数
    L2-norm=np.sqrt(np.sum(np.square(x), axis=axis))
    L1-norm=np.sum(np.abs(x), axis=axis))
    """
    y = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return y


def l2(x, axis=0):
    """L-2范数"""
    y = np.sum(np.square(x), axis=axis)
    return y


def l2_norm(x, axis=0):
    """L-2范数"""
    y = np.sqrt(l2(x, axis=axis))
    return y


def l1(x, axis=0):
    """L-1范数:"""
    y = np.sum(np.abs(x), axis=axis)
    return y


def L1_loss(y_true, y_pre):
    return np.sum(np.abs(y_true - y_pre))


def L2_loss(y_true, y_pre):
    return np.sum(np.square(y_true - y_pre))


def mean_squared_error(y_true, y_pre):
    """MSE(Mean Squared Error)平均平方误差(L2)"""
    l2 = np.sum(np.square(y_true - y_pre))
    return l2 / y_true.size


def mean_absolute_error(y_true, y_pre):
    """MAE(Mean Absolute Error)平均绝对差值(L1),也等于MAD(Mean Absolute Difference)"""
    l1 = np.sum(np.abs(y_true - y_pre))
    return l1 / y_true.size


def mean(data):
    return np.mean(data)


def var(data):
    # 求方差
    return np.var(data)


def std(data):
    # 求标准差
    return np.std(data, ddof=1)


def load_data(data_path):
    return np.load(data_path)


def save_bin(data, bin_file, dtype="double"):
    """
    https://www.cnblogs.com/yaos/p/12105108.html
    C++int对应Python np.intc
    C++float对应Python np.single
    C++double对应Python np.double
    :param data:
    :param bin_file:
    :param dtype:
    :return:
    """
    data = data.astype(np.double)
    data.astype(dtype).tofile(bin_file)


def load_bin(bin_file, shape=None, dtype="double"):
    """
    ==================python load bin data ================
    bin_file = "data.bin"
    shape = (2, 5)
    data1 = np.arange(10, 20).reshape(shape)
    save_bin(data1, bin_file)
    data2 = load_bin(bin_file, shape)
    print(data1)
    print(data2)
    ===================C++ load bin data ===================
    #include <iostream>
    #include <fstream>
    using namespace std;
    int main()
    {
      int row=2;
      int col=5;
      double fnum[row][col] = {0};
      ifstream in("bin/data.bin", ios::in | ios::binary);
      in.read((char *) &fnum, sizeof fnum);
      cout << in.gcount() << " bytes read\n";
      // show values read from file
      for(int i=0; i<row; i++){
          for(int j=0;j<col;j++){
                cout << fnum[i][j] << ",";
          }
           std::cout<<endl;
      }
      in.close();
      return 0;
    }
    >>result:
    80 bytes read
    10,11,12,13,14,
    15,16,17,18,19,
    =======================================================
    :param bin_file:
    :param dtype:
    :return:
    """
    data = np.fromfile(bin_file, dtype=dtype)
    if shape:
        data = np.reshape(data, shape)
    return data


def get_mat_argmax(data):
    """
    寻找置信度最高的那个下标

     data= [[30 54 25]
            [40 40 89]
            [29 36 25]]
    index,v = get_mat_argmax(data)
    ===============
    index = [1, 2]
      v  = 89
    :param data:
    :return: index=[y,x]
             v= max val
    """

    cols = data.shape[1]
    loc = np.argmax(data)
    y = loc // cols
    x = loc % cols
    v = data[y, x]
    return [y, x], v


def get_mat_argmin(data):
    """
    寻找置信度最高的那个下标

     data= [[30 54 25]
            [40 40 89]
            [29 36 25]]
    index,v = get_mat_argmax(data)
    ===============
    index = [1, 2]
      v  = 89
    :param data:
    :return: index=[y,x]
             v= max val
    """

    cols = data.shape[1]
    loc = np.argmin(data)
    y = loc // cols
    x = loc % cols
    v = data[y, x]
    return [y, x], v


def rmse(data1, data2):
    '''
    均方差
    :param predictions:
    :param targets:
    :return:
    '''
    return np.sqrt(((data1 - data2) ** 2).mean())


def get_error(data1, data2):
    """
    MSE（均方误差）、RMSE （均方根误差）、MAE （平均绝对误差）
    :param data1:
    :param data2:
    :return:
    """
    mse = metrics.mean_squared_error(data1, data2)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(data1, data2)
    return mse, rmse, mae


def get_topK(data, k=1, axis=-1, reverse=False):
    """
    选取Top5: contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    多维数组排序
    Args:
        data: 多维数组
        k: 取数
        axis: 轴维度
        reverse: 是否倒序
    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    data = np.asarray(data)
    if reverse:
        partition_index = np.take(np.argpartition(data, kth=k, axis=axis), range(0, k), axis)
    else:
        # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
        # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
        # kth绝对值越小，分区排序效果越明显
        axis_length = data.shape[axis]
        partition_index = np.take(np.argpartition(data, kth=-k, axis=axis),
                                  range(axis_length - k, axis_length), axis)
    top_scores = np.take_along_axis(data, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if not reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_k_value = np.take_along_axis(top_scores, sorted_index, axis)
    top_k_index = np.take_along_axis(partition_index, sorted_index, axis)
    return top_k_value, top_k_index


class Preprocessing(object):
    """
    参考：https://www.cnblogs.com/jojo123/p/6770340.html
    PS：  输入数据x表示样本特征矩阵，其中每行是一个样本，每列是一个相同的特征属性
          一般数据处理都是针对属性特征(列)进行操作的
    """

    @staticmethod
    def scale(x, **kwargs):
        """
        功能：标准化:(x-mean)/std
        说明：将数据按其属性(按列进行)减去其均值，然后除以其方差。
              最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
        """
        y = preprocessing.scale(x, **kwargs)
        # mean = x.mean(axis=0)  # calculate mean
        # std = x.std(axis=0)  # calculate variance
        # y = (x - mean) / std  # standardize X
        return y

    @staticmethod
    def minmax_scaler(x, feature_range=(0, 1), **kwargs):
        """
        功能：最大最小值归一化
        说明：将属性缩放到一个指定的最大值和最小值(通常是1-0)之间
        """
        scaler = preprocessing.MinMaxScaler(feature_range=feature_range, **kwargs)
        y = scaler.fit_transform(x)
        # y = Preprocessing.minmax_normalization(x, omin=feature_range[0], omax=feature_range[1])
        return y

    @staticmethod
    def minmax_normalization(x, omin=0.0, omax=1.0, imin=None, imax=None):
        """
        功能：NORMALIZATION 将数据x归一化到任意区间[ymin,omax]范围的方法
        :param x:  输入参数x：需要被归一化的数据,numpy
        :param omin: 输入参数omin：归一化的区间[omin,omax]下限
        :param omax: 输入参数ymax：归一化的区间[omin,omax]上限
        :param imin: 输入参数imin的最小值
        :param imax: 输入参数ymax的最大值
        :return: 输出参数y：归一化到区间[omin,omax]的数据
        """
        imax = imax if imax is not None else np.max(x, axis=0)  # 计算每列最大值
        imin = imin if imin is not None else np.min(x, axis=0)  # 计算每列最小值
        y = (omax - omin) * (x - imin) / (imax - imin) + omin
        return y

    @staticmethod
    def normalization(x, norm='l2', **kwargs):
        """
        功能：正则化
        说明：对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，
              这样处理的结果是使得每个处理后样本的p-范数(l1-norm,l2-norm)等于1(每行和等于1)
        """
        y = preprocessing.normalize(x, norm=norm, **kwargs)
        # norms = np.sqrt(np.sum(np.square(x), axis=1))
        # y = x / norms[:, np.newaxis]
        return y

    @staticmethod
    def feature_norm(x, axis=1):
        """
        特征归一化
        :param x: 输入二维矩阵(N,embedding-size)，每行是一个样本，样本特征维度为embedding-size
        :param axis:
        :return:
        """
        y = x / np.linalg.norm(x, axis=axis, keepdims=True)
        return y


if __name__ == "__main__":
    from pybaseutils import numpy_utils

    y = np.array([[0., 0.1, 0, 1], [0, 0.1, 1, 1], [0, 0.1, 2, 1]])
    x = np.array([[0, 0.1, 1, 1], [0, 0.1, 2, 1], [0., 0.1, 0, 1]])
    x1 = feature_norm(x)
    x2 = torch_norm(x).numpy()
    print(l2(x1, axis=-1))
    print(l2(x2, axis=-1))
    y = feature_norm(y)
    print(feature_matching(x1, y))
