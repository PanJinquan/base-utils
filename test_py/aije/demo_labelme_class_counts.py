"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-10-09 20:29:58
# @Brief  :
# --------------------------------------------------------
"""
from pybaseutils.dataloader import parser_labelme

if __name__ == "__main__":
    # TODO 统计labelme数据集每个类别的样本数量
    # example_for_segment()
    # example_for_keypoints()
    datasets = [
        '/home/PKing/edudata/dataset/AICQ/dataset/data-2026-10-06/images',
        '/home/PKing/edudata/dataset/AICQ/dataset/data-2026-09-30/images',
    ]
    counts = parser_labelme.get_class_counts(datasets)
    print(counts)
