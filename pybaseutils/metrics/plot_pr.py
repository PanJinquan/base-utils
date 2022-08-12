# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-08-11 16:11:13
    @Brief  : https://www.cnblogs.com/volcao/p/9401304.html
    @Brief  : https://blog.csdn.net/u012370185/article/details/98244530
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def precision_recall_auc(true_labels, pred_scores, thresh=(0.0, 1.0), step=0.1, use_sklearn=False):
    """
    计算precision, recall, auc
    对于二分类问题，可以直接调用sklearn.metrics的precision_recall_curve()
    :param true_labels: 真实样本的的正负标签
    :param pred_scores: 预测的分数
    :param thresh: 阈值范围
    :param step: 阈值间隔
    :param use_sklearn: False:自定义进行计算PR，True:使用sklearn库进行计算PR
    :return: precision, recall, auc, thresholds
            precision : array, shape = [n_thresholds + 1]
                Precision values such that element i is the precision of
                predictions with score >= thresholds[i] and the last element is 1.

            recall : array, shape = [n_thresholds + 1]
                Decreasing recall values such that element i is the recall of
                predictions with score >= thresholds[i] and the last element is 0.
    """
    true_labels, pred_scores = np.asarray(true_labels), np.asarray(pred_scores)
    if use_sklearn:
        precision, recall, threshold = metrics.precision_recall_curve(true_labels, pred_scores)
        precision, recall = precision[:-1], recall[:-1]
    else:
        precision, recall = [], []
        threshold = np.arange(thresh[0], thresh[1], step, dtype=np.float32)
        for thresh in threshold:
            label = np.array(pred_scores >= thresh - 1e-6, dtype=np.int)
            p = metrics.precision_score(true_labels, label)
            r = metrics.recall_score(true_labels, label)
            # print("{:.2f}={},r={},p={},".format(thresh, label, r, p))
            precision.append(p)
            recall.append(r)
    auc = metrics.auc(recall, precision)
    return precision, recall, auc, threshold


def plot_precision_recall_curve(true_labels, pred_scores, thresh=(0.0, 1.0), step=0.1, use_sklearn=False, vis=True):
    """
    计算precision, recall, auc并绘制PR曲线(x=Recall,y=Precision)
    对于二分类问题，可以直接调用sklearn.metrics的precision_recall_curve()
    :param true_labels: 真实样本的的正负标签
    :param pred_scores: 预测的分数
    :param thresh: 阈值范围
    :param step: 阈值间隔
    :param use_sklearn: False:自定义进行计算PR，True:使用sklearn库进行计算PR
    :return: precision, recall, auc
    """
    precision, recall, auc, threshold = precision_recall_auc(true_labels, pred_scores, thresh=thresh,
                                                             step=step, use_sklearn=use_sklearn)
    if vis:
        print("Threshold:{}".format(threshold.tolist()))
        print("Recall   :{}".format(recall))
        print("Precision:{}".format(precision))
        print("AUC(PR)  :{}".format(auc))
        plt_pr_curve([precision], [recall], [auc], names=[""])
        plt_pr_threshold(precision, recall, threshold, title="Precision-Recall-Threshold")
    return precision, recall, auc


def plt_pr_curve(p_list, r_list, auc_list, names, grid=True):
    """
    绘制PR曲线(x=Recall,y=Precision)
    :param p_list: Precision List
    :param r_list: Recall List
    :param auc_list: AUC List
    :param names:曲线名称
    :return:
    """
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    for r, p, auc, c, n in zip(r_list, p_list, auc_list, colors, names):
        # Recall为横坐标，Precision为纵坐标做曲线
        plt.plot(r, p, color=c, lw=lw, label='{} AUC={:.3f}'.format(n, auc))
    # 绘制直线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'weight': 'normal', 'size': 20}
    plt.xlabel('Recall', font)
    plt.ylabel('Precision', font)
    plt.title('PR curve', font)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(grid)  # 显示网格
    plt.show()


def plt_pr_threshold(precision, recall, threshold, title, grid=True):
    """
    绘制PR-threshold曲线
    :param precision: Precision
    :param recall: Recall
    :param threshold
    :param title:曲线名称
    :return:
    """
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    plt.plot(threshold, precision, color=colors[0], lw=lw, label='Precision')
    plt.plot(threshold, recall, color=colors[1], lw=lw, label='Recall')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'weight': 'normal', 'size': 20}
    plt.xlabel('thresholds', font)
    plt.ylabel('', font)
    plt.title(title, font)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(grid)  # 显示网格;
    plt.show()


if __name__ == "__main__":
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 真实类别
    pred_scores = [0.1, 0.2, 0.7, 0.3, 0.4, 0.5, 0.2, 0.7, 0.8, 0.9]  # 预测类别分数
    precision, recall, AUC = plot_precision_recall_curve(true_labels, pred_scores)
