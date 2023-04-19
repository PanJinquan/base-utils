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
    data =  data[["image_id","label"]]
    data =  get_rows_by_keys(df, ["image_id", "label"]).values
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
