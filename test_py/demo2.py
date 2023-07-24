# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import cv2


# 如算法返回angle=5,则需要对原图进行反向旋转对应的角度，可如下调用：
# image_rotation(image, angle=-5)
def image_rotation(image, angle):
    """实现图像旋转"""
    h, w = image.shape[:2]
    center = (w / 2., h / 2.)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, mat, dsize=(w, h))
    return image
