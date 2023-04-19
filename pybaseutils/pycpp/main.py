# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-25 16:39:57
    @Brief  : ctypes例子：https://blog.csdn.net/milkhoko/article/details/119326249
              ctypes例子：https://blog.csdn.net/u012819437/article/details/89642312
              ctypes指针: https://blog.csdn.net/weixin_39636057/article/details/109948238
"""
import ctypes
import cv2
import numpy as np

# 通过ctypes.CDLL可以载入动态链接库，这个载入方法是全平台通用的，建议使用这种
# CTLib = ctypes.CDLL("build/libCTLib.so")
CTLib = ctypes.CDLL("cmake-build-debug/libCTLib.so")


class CTImage(ctypes.Structure):
    _fields_ = [
        ('rows', ctypes.c_int),  # H
        ('cols', ctypes.c_int),  # W
        ('dims', ctypes.c_int),  # channels
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ]

    @staticmethod
    def instance(image: np.ndarray):
        """
        获得实例对象
        """
        rows, cols, channels = image.shape  # (h,w,d)
        info = CTImage(rows, cols, channels, image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        return info


def test_ct_imread(filename: str, vis=True):
    """
    C++: uchar *ct_imread(char *filename, int *rows, int *cols, int *channels)
    :param filename:
    :return:
    """
    # 声明输入参数(rows,cols,channels)为int类型的指针参数:
    rows = ctypes.c_int(0)
    cols = ctypes.c_int(0)
    channels = ctypes.c_int(0)
    ct_imread = CTLib.ct_imread
    # 声明函数输入类型：(char *filename, int *rows, int *cols, int *channels)
    # ct_imread.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    ct_imread.argtypes = (ctypes.POINTER(ctypes.c_char),
                          ctypes.POINTER(ctypes.c_int),
                          ctypes.POINTER(ctypes.c_int),
                          ctypes.POINTER(ctypes.c_int))
    # string_p = ctypes.create_string_buffer(1024, '\0')
    # 声明函数返回类型：uchar *
    # ct_imread.restype = ctypes.c_void_p
    ct_imread.restype = ctypes.POINTER(ctypes.c_ubyte)
    # 调用函数：str类型需要转为bytes对象，即encode();byref指明参数传递时为引用传递，对应着C语言的指针传递,
    dst = ct_imread(filename.encode(), ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(channels))
    # 通过value取地址值
    cols, rows, channels = cols.value, rows.value, channels.value
    # 将读取的图片转为numpy进行显示
    dst = ctypes.string_at(dst, cols * rows * channels)  # string_at(c_str_p) # 获取内容
    image = np.frombuffer(dst, np.uint8)
    image = np.reshape(image, newshape=(rows, cols, channels))
    if vis:
        cv2.imshow('imread', image)
        cv2.waitKey(0)
    return image


def test_ct_blur(src: np.ndarray, vis=True):
    """
    :param src:
    :return:
    """
    rows, cols, channels = src.shape
    dst = np.zeros_like(src)
    ct_blur = CTLib.ct_blur
    # 声明函数输入类型
    ct_blur.argtypes = (ctypes.POINTER(ctypes.c_ubyte),
                        ctypes.POINTER(ctypes.c_ubyte),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int)
    src_p = src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_p = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    # src_p = src.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    # dst_p = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    ct_blur(src_p, dst_p, rows, cols, channels)
    if vis:
        cv2.imshow('blur', dst)
        cv2.waitKey(0)
    return dst


def test_struct(src: np.ndarray, vis=True):
    """
    测试ctypes结构体
    :param src:
    :param vis:
    :return:
    """
    dsize = (400, 200)  # (w,h)
    dst = np.zeros(shape=(dsize[1], dsize[0], 3), dtype=np.uint8)
    src_obj = CTImage.instance(src)
    dst_obj = CTImage.instance(dst)
    CTLib.ct_resize(ctypes.byref(src_obj), ctypes.byref(dst_obj), dsize[0], dsize[1])
    CTLib.printCTImage(src_obj)
    CTLib.printCTImage(dst_obj)
    if vis:
        cv2.imshow('struct-dst', dst)
        cv2.waitKey(0)
    return dst


if __name__ == "__main__":
    filename = "test.png"
    image = test_ct_imread(filename)
    image = test_ct_blur(image)
    image = test_struct(image)
