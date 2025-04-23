# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-07-30 20:13:59
"""
import os
import numpy as np
import pandas as pd


def read_csv(filename, sep=";"):
    """
    :param filename:
    :param sep: 分隔符
    :return:
    """
    file = pd.read_csv(filename, sep=sep)
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


def save_csv(filename, df: pd.DataFrame, rows=True):
    """
    :param filename:
    :param df:
    :param rows:
    :return:
    """
    if rows is None: rows = True
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=rows, sep=',', header=True)


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


def data2df(data, cols=None, rows=None, file=None) -> pd.DataFrame:
    """
    将data数据转为pd.DataFrame
    :param data: 表单数据
    :param cols: (columns)表单列名称
    :param rows: (index)表单行名称
    :param file:
    :return: pd.DataFrame
    """
    df = pd.DataFrame(data, index=rows, columns=cols)  # 生成6行4列位置
    if file: save_csv(file, df, rows=rows)
    return df


def dict2df(data: dict, cols=None, T=False, file=None):
    """
    :param data: 表单数据
    :param cols: (columns)表单列名称
    :param T: 是否转置表单
    :return: pd.DataFrame
    """
    if T:
        df = pd.DataFrame.from_dict(data, columns=cols)  # 键按照列进行转换
    else:
        df = pd.DataFrame.from_dict(data, columns=cols, orient='index')  # 键按照行进行转换
    if file: save_csv(file, df, rows=True)
    return df


if __name__ == "__main__":
    # TODO
    cols = ["C1", "C2"]
    rows = None
    data = np.arange(0, 6).reshape(3, 2)
    df = data2df(data, cols, rows, file="data1.csv")
    print(df)

    # TODO
    data = {
        'name1': ["A0", "A1", "A2"],
        'name2': ["B0", "B1", "B2"],
        'name3': ["C0", "C1", "C2"]
    }
    cols = ["C1", "C2", "C3"]
    df = dict2df(data, cols=cols, file="data2.csv")
