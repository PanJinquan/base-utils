# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import cv2
import random
import numpy as np
from pybaseutils import time_utils, image_utils


@time_utils.performance("gaussian_noise")
def gaussian_noise(image, mean=0.1, sigma=0.1):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output


@time_utils.performance("salt_pepper_noise")
def salt_pepper_noise(image: np.ndarray, prob=0.01):
    """
    添加椒盐噪声，该方法需要遍历每个像素，十分缓慢
    :param image:
    :param prob: 噪声比例
    :return:
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.random() < prob:
                image[i][j] = 0 if random.random() < 0.5 else 255
            else:
                image[i][j] = image[i][j]
    return image


@time_utils.performance("fast_salt_pepper_noise")
def fast_salt_pepper_noise(image: np.ndarray, prob=0.02):
    """
    随机生成一个0~1的mask，作为椒盐噪声
    :param image:图像
    :param prob: 椒盐噪声噪声比例
    :return:
    """
    image = add_uniform_noise(image, prob * 0.51, vaule=255)
    image = add_uniform_noise(image, prob * 0.5, vaule=0)
    return image


def add_uniform_noise(image: np.ndarray, prob=0.05, vaule=255):
    """
    随机生成一个0~1的mask，作为椒盐噪声
    :param image:图像
    :param prob: 噪声比例
    :param vaule: 噪声值
    :return:
    """
    h, w = image.shape[:2]
    shape = [int(random.uniform(h // 8, h // 4)), int(random.uniform(w // 8, w // 4))]
    print(shape)
    noise = np.random.uniform(low=0.0, high=1.0, size=shape).astype(dtype=np.float32)  # 产生高斯噪声
    noise = cv2.resize(noise, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(shape=(h, w), dtype=np.uint8) + vaule
    index = noise > prob
    mask = mask * (~index)
    output = image * index[:, :, np.newaxis] + mask[:, :, np.newaxis]
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output


def cv_show_image(title, image, use_rgb=True, delay=0):
    """
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入是否是RGB图像
    :param use_rgb: True:输入image是RGB的图像, False:返输入image是BGR格式的图像
    :return:
    """
    img = image.copy()
    if img.shape[-1] == 3 and use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    # cv2.namedWindow(title, flags=cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(delay)
    return img


if __name__ == "__main__":
    test_file = "test.png"
    image = cv2.imread(test_file)
    image = image_utils.resize_image(image, size=(224, 224))
    for i in range(100):
        prob = 0.02
        out1 = gaussian_noise(image.copy())
        out2 = salt_pepper_noise(image.copy(), prob=prob)
        out3 = add_uniform_noise(image.copy(), prob=prob)
        print("----" * 10)
        cv_show_image("image", image, use_rgb=False, delay=1)
        cv_show_image("gaussian_noise", out1, use_rgb=False, delay=1)
        cv_show_image("salt_pepper_noise", out2, use_rgb=False, delay=1)
        cv_show_image("add_uniform_noise", out3, use_rgb=False, delay=0)
        cv2.line()
