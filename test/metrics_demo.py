# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-26 15:55:30
    @Brief  :
"""
from pybaseutils import classification_report

true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 真实类别
pred_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # 预测类别
target_names = ["负样本", "正样本"]
# 计算混淆矩阵
conf_matrix = classification_report.get_confusion_matrix(true_labels, pred_labels,
                                                         target_names=target_names,
                                                         filename="./confusion_matrix.csv",
                                                         normalization=False,
                                                         plot=True,
                                                         title="Confusion Matrix")
# 计算Accuracy、Precision、Recall、F1-Score
report = classification_report.get_classification_report(true_labels, pred_labels,
                                                         target_names=target_names,
                                                         output_dict=False)

print(report)


def binary_class_demo():
    """
    计算二分类的precision,recall,f1-score
    :return:
    """
    true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    pred_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    target_names = ["other", "个"]
    report = classification_report.get_classification_report(true_labels, pred_labels, target_names=target_names,
                                                             output_dict=False)
    print(report)
    filename = "./confusion_matrix.csv"
    classification_report.get_confusion_matrix(true_labels, pred_labels, target_names=target_names, filename=filename,
                                               normalization=False, plot=True, title="Confusion Matrix")


def multi_class_demo():
    """
    计算多分类的precision,recall,f1-score
    :return:
    """
    true_labels = [0, 1, 2, 3, 3, 1, 1]
    pred_labels = [0, 1, 2, 2, 2, 1, 0]
    target_names = ["A", "B", "C", "D"]
    # 计算多分类的precision,recall,f1-score
    out_result = classification_report.get_classification_report(true_labels,
                                                                 pred_labels,
                                                                 target_names=target_names,
                                                                 output_dict=False)
    print(out_result)
    classification_report.get_confusion_matrix(true_labels, pred_labels,
                                               target_names=target_names,
                                               normalization=False,
                                               plot=True,
                                               title="Confusion Matrix")


if __name__ == "__main__":
    pass
    # binary_class_demo()
    # multi_class_demo()
