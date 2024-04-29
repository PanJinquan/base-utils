# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-07-04 14:43:26
    @Brief  : pip install nibabel
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    nii_file = "/home/PKing/nasdata/dataset/tmp/challenge/脑PET图像分析和疾病预测/脑PET图像分析和疾病预测挑战赛公开数据/Test/1.nii"
    img = nib.load(nii_file)
    print(img.shape)  # shape(240, 240, 155)
    print(img.header['db_name'])
    width, height, queue, _ = img.dataobj.shape  # 由文件本身维度确定，可能是3维，也可能是4维
    # print("width",width)  # 240
    # print("height",height) # 240
    # print("queue",queue)   # 155
    # nib.viewers.OrthoSlicer3D(img.dataobj).show()

    num = 1
    for i in range(0, queue, 1):
        img_arr = img.dataobj[:, :, i,0]
        plt.subplot(queue//4+1, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()
