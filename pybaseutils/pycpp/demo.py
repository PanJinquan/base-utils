from ctypes import *
import cv2  # opencv读取的格式是BGR
import numpy as np
import ctypes

# 通过ctypes.CDLL可以载入动态链接库，这个载入方法是全平台通用的，建议使用这种
# CTLib = ctypes.CDLL("build/libCTLib.so")
CTLib = ctypes.CDLL("cmake-build-debug/libCTLib.so")


class ImageInfo(ctypes.Structure):
    _fields_ = [
        ('rows', ctypes.c_int),
        ('cols', ctypes.c_int),
        ('channels', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ]

    @staticmethod
    def instance(image: np.ndarray):
        """
        获得实例对象
        """
        rows, cols, channels = image.shape
        info = ImageInfo(rows, cols, channels, image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        return info


if __name__ == "__main__":
    filename = "test.png"
    src = cv2.imread(filename)
    dsize = (200, 200)  # (w,h)
    cv2.imshow('src', src)
    rows, cols, channels = src.shape
    dst = np.zeros(shape=(dsize[1], dsize[0], 3), dtype=np.uint8)
    src_obj = ImageInfo.instance(src)
    dst_obj = ImageInfo.instance(dst)
    CTLib.ct_resize(ctypes.byref(src_obj), ctypes.byref(dst_obj), dsize[1], dsize[0])
    CTLib.printImageInfo(src_obj)
    CTLib.printImageInfo(dst_obj)
    # cv2.imshow('dst', src)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
