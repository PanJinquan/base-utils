# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-06-04 09:43:22
    @Brief  :
"""
import os
import random
from pybaseutils import file_utils, image_utils
from basetrainer.metric.eval_tools import roc, pr


class FaceRecognition(object):
    def __init__(self):
        pass

    def test_pair_files(self, image_dir, filename):
        pair_data = file_utils.read_data(filename, split=" ")
        label = []
        score = []
        for file1, file2, issame in pair_data:
            file1 = os.path.join(image_dir, file1)
            file2 = os.path.join(image_dir, file2)
            s = random.uniform(0, 1.0)
            label.append(issame)
            score.append(s)
        # TODO 测试
        fpr, tpr, roc_auc, thresh_list, best_idx = roc.get_roc_curve(y_true=label,
                                                                     y_score=score,
                                                                     invert=False,
                                                                     plot_roc=True)
        thresh = thresh_list[best_idx]
        # print("thresh_list :{}".format(thresh_list))
        print("roc_auc     :{}".format(roc_auc))
        print("optimal_idx :{},best_threshold :{} ".format(best_idx, thresh))
        pred = [int(s > thresh) for s in score]
        precision, recall, acc = pr.get_precision_recall_acc(label, pred, average="binary")
        precision, recall, AUC = pr.plot_classification_pr_curve(label, pred)
        print("precision   :{}".format(precision))
        print("recall      :{}".format(recall))
        print("acc         :{}".format(acc))
        print("AUC         :{}".format(AUC))


if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/人脸识别数据/X4_Face50/trainval"
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/人脸识别数据/X4_Face50/x4_pair_data.txt"
    FR = FaceRecognition()
    FR.test_pair_files(image_dir=image_dir, filename=filename)
