# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : plot_utils.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-13 16:30:10
"""
# 导入需要用到的库
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


def count_bin(x, bin_ranges, num_bin=10, norm=True):
    """
    统计x在范围bin_ranges出现的频率
    Example:
        x = np.arange(0, 1, 0.1)
        y = (0, 0.001, 0.11, 0.15, 0.11, 0.11, 0.2, 0.3, 0.31)
        count = count_bin(y, bin_ranges=[0, 1.0], num_bin=len(x))
        plot_bar_text(list(count.keys()), list(count.values()))
    :param x:
    :param bin_ranges: [min,max]
    :param num_bin: 均匀分割bin个数
    :param norm: 是否转换概率密度
    :return:
    """
    bin_step = (bin_ranges[1] - bin_ranges[0]) / num_bin
    bin_labels = np.arange(bin_ranges[0], bin_ranges[1], bin_step)
    labels = np.digitize(x, bin_labels)
    count = {c: 0 for c in bin_labels}
    for i in range(len(labels)):
        l = labels[i]
        c = bin_labels[l]
        count[c] += 1
    if norm:
        s = sum(count.values())
        count = {k: v / s for k, v in count.items()}
    return count


def plot_bar_text(x, y, xlabel="X", ylabel="Y", title="bar", bin_width=0.1):
    """绘制条形直方图"""
    rects = plt.bar(x=x, height=y, width=bin_width, align="center", yerr=0.000001)
    for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 3., 1.03 * h, "{:3.1f}".format(h))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(True)  # 显示网格;
    plt.savefig('out.png')
    plt.show()


def plot_bar(x_data, y_data, title, xlabel, ylabel, isshow=False):
    # 准备数据
    # 用 Matplotlib 画条形图
    plt.bar(x_data, y_data)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10,
            }
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)

    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(True)  # 显示网格;
    plt.savefig('out.png')
    if isshow:
        plt.show()


def plot_multi_line(x_data_list, y_data_list, line_names=None, title="", xlabel="", ylabel=""):
    # 绘图
    # plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    xlim_max = 0
    ylim_max = 0

    xlim_min = 0
    ylim_min = 0
    if not line_names:
        line_names = " " * len(x_data_list)
    for x, y, color, line_name in zip(x_data_list, y_data_list, colors, line_names):
        plt.plot(x, y, color=color, lw=lw, label=line_name)  # 假正率为横坐标，真正率为纵坐标做曲线
        if xlim_max < max(x):
            xlim_max = max(x)
        if ylim_max < max(y):
            ylim_max = max(y)
        if xlim_min > min(x):
            xlim_min = min(x)
        if ylim_min > min(y):
            ylim_min = min(y)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线
    x_deta = xlim_max - xlim_min
    y_deta = ylim_max - ylim_min
    plt.xlim([xlim_min - 0.01 * x_deta, xlim_max + 0.1 * x_deta])
    plt.ylim([ylim_min - 0.01 * y_deta, ylim_max + 0.1 * y_deta])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)

    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(True)  # 显示网格;
    plt.show()


def plot_skew_kurt(data, name="Title"):
    """
    https://blog.csdn.net/u012735708/article/details/84750295
    计算偏度(skew)和峰度(kurt)
    :return:
    """
    import pandas as pd
    plt.figure(figsize=(10, 10))
    skew = pd.Series(data).skew()
    kurt = pd.Series(data).kurt()
    info = 'skew={:.4f},kurt={:.4f},mean:{:.4f}'.format(skew, kurt, np.mean(data))  # 标注
    info = "{}:\n{}".format(name, info)
    plt.title(info)
    print(info)
    plt.hist(data, 100, facecolor='r', alpha=0.9)
    plt.grid(True)
    plt.show()


def demo(image1, image2):
    fig = plt.figure(2)  # 新开一个窗口
    # fig1
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1)
    ax1.set_title("image1")

    # fig2
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image2)
    ax2.set_title("image2")
    plt.show()


def demo_for_skew_kurt():
    """
    https://blog.csdn.net/u012735708/article/details/84750295
    计算偏度(skew)和峰度(kurt)
    :return:
    """
    import numpy as np
    data = list(np.random.randn(10000))
    plot_skew_kurt(data)


if __name__ == "__main__":
    import cv2

    # image_path="/media/dm/dm1/git/python-learning-notes/dataset/test_image/1.jpg"
    # image=cv2.imread(image_path)
    # image1=cv2.resize(image,dsize=(100,100))
    # demo(image, image1)
    demo_for_skew_kurt()
