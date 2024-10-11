# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : pandas_tools.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-07-30 20:13:59
"""
import numpy as np
import pandas as pd


def read_csv(filename, usecols=None):
    """
    :param filename:
    :param usecols:
    :return:
    """
    file = pd.read_csv(filename, usecols=usecols)
    df = pd.DataFrame(file)
    return df


def get_rows_by_keys(df, keys=[]) -> pd.DataFrame:
    """
    data =  data[["image_ids","label"]]
    data =  get_rows_by_keys(df, ["image_ids", "label"]).values
    :param df:
    :param keys:
    :return:
    """
    data = df[keys]
    return data


def df2list(df):
    """pandas.values转为list"""
    list_ = df.values.tolist()
    return list_


def save_csv(filename, df, save_index=True):
    """
    :param filename:
    :param df:
    :param save_index:
    :return:
    """
    df.to_csv(filename, index=save_index, sep=',', header=True)


def print_info(class_name, labels):
    """
    :param class_name:
    :param labels:
    :return:
    """
    # index =range(len(class_name))+1
    index = np.arange(0, len(class_name)) + 1
    columns = ['class_name', 'labels']
    content = np.array([class_name, labels]).T
    df = pd.DataFrame(content, index=index, columns=columns)  # 生成6行4列位置
    print(df)  # 输出6行4列的表格
    save_csv("my_test.csv", df)


def construct_pd(index, columns_name, content, filename=None):
    df = pd.DataFrame(content, index=index, columns=columns_name)  # 生成6行4列位置
    save_index = True
    if not index:
        save_index = False
    if filename is not None:
        save_csv(filename, df, save_index=save_index)
    return df


def dict2pd(data: dict, T=False):
    """
    :param data:
    :param T:
    :return:
    """
    if T:
        df = pd.DataFrame.from_dict(data)  # 键按照列进行转换
    else:
        df = pd.DataFrame.from_dict(data, orient='index')  # 键按照行进行转换
    return df


if __name__ == "__main__":
    class_name = ['C1', 'C2', 'C3']
    labels = [100, 200, 300]
    # print_info(class_name, labels)
    # index = [1, 2, 3, 4, 5, 8]
    index = None
    columns_name = ["A", "B"]
    content = np.arange(0, 12).reshape(6, 2)
    filename = "my_test.csv"
    df = construct_pd(index, columns_name, content, filename)
    print(df)

    data = {
        'name1': ["A0", "A1", "A2"],
        'name2': ["B0", "B1", "B2"],
        'name3': ["C0", "C1", "C2"]
    }
    df = dict2pd(data)
    save_csv("data.csv", df)
