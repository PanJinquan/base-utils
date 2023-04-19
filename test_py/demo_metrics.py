# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-26 15:55:30
    @Brief  :
"""
from pybaseutils.metrics import class_report


def binary_class_example():
    """
    计算二分类的precision,recall,f1-score以及混淆矩阵
    :return:
    """
    true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 真实类别
    pred_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # 预测类别
    target_names = ["负样本", "正样本"]
    # 计算混淆矩阵
    conf_matrix = class_report.get_confusion_matrix(true_labels, pred_labels,
                                                    target_names=target_names,
                                                    filename="./confusion_matrix.csv",
                                                    normalization=False,
                                                    plot=False,
                                                    title="Confusion Matrix")
    # 计算Accuracy、Precision、Recall、F1-Score
    report = class_report.get_classification_report(true_labels, pred_labels,
                                                    target_names=target_names,
                                                    output_dict=False)

    print(report)


def multi_class_example():
    """
    计算多分类的precision,recall,f1-score以及混淆矩阵
    :return:
    """
    true_labels = [0, 1, 2, 3, 3, 1, 1]
    pred_labels = [0, 1, 2, 2, 2, 1, 0]
    target_names = ["A", "B", "C", "D"]
    # 计算多分类的precision,recall,f1-score
    out_result = class_report.get_classification_report(true_labels,
                                                        pred_labels,
                                                        target_names=target_names,
                                                        output_dict=True)
    print(out_result)
    class_report.get_confusion_matrix(true_labels, pred_labels,
                                      target_names=target_names,
                                      normalization=False,
                                      plot=True,
                                      title="Confusion Matrix")


if __name__ == "__main__":
    from pybaseutils.metrics import plot_roc, plot_pr

    # 真实类别
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # 预测类别分数
    pred_scores = [0.1, 0.2, 0.7, 0.3, 0.4, 0.5, 0.2, 0.7, 0.8, 0.9]
    # 绘制PR曲线
    # plot_pr.plot_precision_recall_curve(true_labels, pred_scores)
    # 绘制ROC曲线
    # plot_roc.plot_roc_curve(true_labels, pred_scores)

    # binary_class_example()
    multi_class_example()
