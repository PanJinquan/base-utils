# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-08-11 16:11:13
    @Brief  : https://www.cnblogs.com/volcao/p/9401304.html
    @Brief  : https://blog.csdn.net/u012370185/article/details/98244530
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_curve(true_labels, pred_scores, vis=True):
    """
    一般情况下，大于阈值时，y_test为1，小于等于阈值时y_test为0, y_test与y_score一一对应,且是正比关系
    当用距离作为y_score的分数时，此时y_test与y_score是反比关系（大于阈值时，y_test为0，小于等于阈值时y_test为1）
    :param true_labels  : 真实值
    :param pred_scores : 预测分数
    :param invert  : 是否对y_test进行反转，当y_test与y_score是正比关系时，invert=False,当y_test与y_score是反比关系时,invert=True
    :param vis: 是否绘制roc曲线
    :return:fpr,
            tpr,
            roc_auc,
            threshold ,阈值点
            optimal_index:最佳截断点,best_threshold = threshold[optimal_idx]获得最佳阈值
    """
    true_labels = np.array(true_labels)
    pred_scores = np.array(pred_scores)
    # Compute ROC curve and ROC area for each class
    # 计算roc
    fpr, tpr, threshold = metrics.roc_curve(true_labels, pred_scores, pos_label=1)
    # 计算auc的值
    roc_auc = metrics.auc(fpr, tpr)
    # 计算最优阈值:最佳截断点应该是tpr高,而fpr低的地方。
    # url :https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    optimal_index = np.argmax(tpr - fpr)
    # best_threshold = threshold[optimal_idx]
    # 绘制ROC曲线
    if vis:
        plt_roc_curve([fpr], [tpr], [roc_auc], names=[""])
        # print("FPR           :{}".format(fpr))
        # print("TPR           :{}".format(tpr))
        # print("Threshold     :{}".format(threshold))
        print("AUC(ROC)      :{}".format(roc_auc))
        print("best_threshold:{} ".format(threshold[optimal_index]))
    return fpr, tpr, roc_auc, threshold, optimal_index


def plt_roc_curve(fpr_list, tpr_list, auc_list, names, grid=True):
    """
    绘制roc曲线
    :param fpr_list:
    :param tpr_list:
    :param auc_list:
    :param names:曲线名称
    :return:
    """
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    for fpr, tpr, auc, c, n in zip(fpr_list, tpr_list, auc_list, colors, names):
        # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr, tpr, color=c, lw=lw, label='{} AUC={:.3f}'.format(n, auc))
        # 绘制直线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'weight': 'normal', 'size': 20}
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)
    plt.title('ROC curve', font)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(grid)  # 显示网格;
    plt.show()


def get_tpr_fpr(fpr, tpr, fixed_fpr=0.01):
    """
    metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.01))
    metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.001))
    metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.0001))
    :param fixed_fpr:<float>
    :return:
    """
    # fpr, tpr, thr = metrics.roc_curve(target, output)
    # fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label=1)
    tpr_filtered = tpr[fpr <= fixed_fpr]
    if len(tpr_filtered) == 0:
        return 0.0
    return tpr_filtered[-1]


def custom_roc_curve(y_true, y_score):
    lw = 2
    plt.figure(figsize=(10, 10))
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    pred_sort = np.sort(y_score)[::-1]  # 从大到小排序
    index = np.argsort(y_score)[::-1]  # 从大到小排序
    y_sort = y_true[index]
    print(y_sort)
    tpr = []
    fpr = []
    thr = []
    for i, item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    print(fpr)
    print(tpr)
    print(thr)
    # 画图
    plt.plot(fpr, tpr, 'k')
    # plt.plot([(0, 0), (1, 1)], 'r--') # y=x
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)

    plt.title('ROC curve')
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.show()


def get_best_accuracy(true_labels, pred_scores, threshold=(0, 1.0, 0.05), vis=False):
    """
    计算最佳准确率和阈值
    :param true_labels:
    :param pred_scores:
    :param vis:
    :return: best_acc, best_th
    """
    true_labels, pred_scores = np.asarray(true_labels), np.asarray(pred_scores)
    threshold = np.arange(threshold[0], threshold[1], threshold[2])
    acc_list = []
    for th in threshold:
        pred_labels = np.asarray(pred_scores > th, dtype=np.int32)
        acc = metrics.accuracy_score(y_true=true_labels, y_pred=pred_labels)
        acc_list.append(acc)
    index = np.argmax(acc_list)
    best_th = threshold[index]
    best_acc = acc_list[index]
    if vis:
        print("best threshold           :{}".format(best_th))
        print("best Accuracy            :{}".format(best_acc))
        plt_curve(x=threshold, y=acc_list, xlabel="threshold", ylabel="Accuracy",
                  line=f"Acc={best_acc:1.3f}", title="", grid=True)
    return best_acc, best_th


def plt_curve(x, y, line="", xlabel="", ylabel="", title="", grid=True):
    """
    绘制PR-threshold曲线
    :param x: Precision
    :param recall: Recall
    :param y
    :param title:曲线名称
    :return:
    """
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    plt.plot(x, y, color=colors[0], lw=lw, label=line)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'weight': 'normal', 'size': 20}
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    plt.title(title, font)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(grid)  # 显示网格;
    plt.show()


if __name__ == "__main__":
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 真实类别
    pred_scores = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 预测类别分数
    # pred_scores = [0.11, 0.21, 0.71, 0.31, 0.41, 0.51, 0.21, 0.71, 0.81, 0.91]  # 预测类别分数
    # plot_roc_curve(true_labels, pred_scores)
    print(get_best_accuracy(true_labels, pred_scores, vis=True))
