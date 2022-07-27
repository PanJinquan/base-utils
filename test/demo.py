# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-26 15:55:30
    @Brief  :
"""
from sklearn import metrics
from pybaseutils.classification_report import get_classification_report, get_confusion_matrix


def binary_class_demo():
    """
    计算二分类的precision,recall,f1-score
    :return:
    """
    true_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    pred_labels = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    target_names = ["other", "个"]
    report = get_classification_report(true_labels, pred_labels, target_names=target_names, output_dict=False)
    prf = metrics.precision_recall_fscore_support(true_labels, pred_labels)
    print(report)
    print("precision_recall_fscore:{}".format(prf))
    filename = "./confusion_matrix.csv"
    get_confusion_matrix(true_labels, pred_labels, target_names=target_names, filename=filename,
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
    out_result = get_classification_report(true_labels, pred_labels, target_names=target_names, output_dict=False)
    print(out_result)
    get_confusion_matrix(true_labels, pred_labels, target_names=target_names, normalization=False, plot=True,
                         title="Confusion Matrix")


if __name__ == "__main__":
    binary_class_demo()
    # multi_class_demo()
