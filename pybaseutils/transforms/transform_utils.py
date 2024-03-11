# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-05 19:38:52
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils.cvutils import corner_utils


def get_target_corner_points(src_pts: np.ndarray):
    """
    根据输入的四个角点，计算其矫正后的目标四个角点,src_pts四个点分布：
        0--(w01)---1
        |          |
      (h03)      (h21)
        |          |
        3--(w23)---2
    :param src_pts: 原始4个角点
    :return: 目标4个角点
    """
    return corner_utils.get_target_points(src_pts=src_pts)


def image_alignment(image, src_pts, dst_pts, out_size=None, method="lstsq"):
    """
    apply affine transform实现对齐图像
    D=M*S or D=H*S
    :param image: input image
    :param src_pts: 原始点S集合(n×2)
    :param dst_pts: 目标点D集合(n×2)
    :param out_size: 变换后输出图像大小
    :param method: lstsq,estimate,affine,homo
    :return:  align_image 对齐后的图像
              M           S->D的变换矩阵(2×3)
              Minv        D->S的逆变换矩阵(2×3)
    """
    if out_size is None:
        xmax = int(max(dst_pts[:, 0]))
        ymax = int(max(dst_pts[:, 1]))
        out_size = (xmax, ymax)
    M = get_transform(src_pts, dst_pts, method=method)
    # 进行仿射变换
    if M.size == 6:  # 如果M是2×3的变换矩阵
        align_image = cv2.warpAffine(image, M=M, dsize=tuple(out_size))
        # Minv = get_transform(dst_pts, src_pts, method=method)
        Minv = get_inverse_matrix(M)
        # Minv = cv2.invertAffineTransform(M)
    else:  # 如果M是3×3的单应矩阵
        align_image = cv2.warpPerspective(image, M, dsize=tuple(out_size))
        Minv = get_transform(dst_pts, src_pts, method=method)
    return align_image, M, Minv


def solve_lstsq(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    通过最小二乘法计算变换矩阵
    np.linalg.lstsq() <==>  cv2.solve()
    :param src_pts:
    :param dst_pts:
    :return: M
    """
    src_pts = np.matrix(src_pts.astype(np.float32))
    dst_pts = np.matrix(dst_pts.astype(np.float32))
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])
    # H, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)
    res, H = cv2.solve(src_pts_, dst_pts_, flags=cv2.DECOMP_QR)
    rank = 3
    if rank == 3:
        M = np.float32([
            [H[0, 0], H[1, 0], H[2, 0]],
            [H[0, 1], H[1, 1], H[2, 1]]
        ])
    elif rank == 2:
        M = np.float32([
            [H[0, 0], H[1, 0], 0],
            [H[0, 1], H[1, 1], 0]
        ])
    return M


def get_transform(src_pts, dst_pts, method="lstsq"):
    """
    获得变换矩阵
    https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd
    findHomography: https://blog.csdn.net/fengyeer20120/article/details/87798638/
    https://blog.csdn.net/weixin_51998427/article/details/130469427
    计算变换矩阵
    仿射变换M：是一个2×3的矩阵，主要由旋转、缩放、平移、斜切等基本变换组成，使用warpAffine进行变换
    单应矩阵H：是一个3×3的矩阵，它可以描述图像平面与图像平面之间的透视变换，使用warpPerspective进行变换
    D=M*S or D=H*S
    :param src_pts: 原始点S集合(n×2)
    :param dst_pts: 目标点D集合(n×2)
    :param method: lstsq,estimate,affine,homo
    :return: M是仿射变换矩阵M(2×3),或者是单应矩阵H(3×3)
    """
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    if method == "lstsq":  # M is 2×3的矩阵
        # 使用最小二乘法计算仿射变换矩阵
        M = solve_lstsq(src_pts, dst_pts)
    elif method == "estimate":  # M is 2×3的矩阵
        # estimateAffine2D()可以用来估计最优的仿射变换矩阵,method = cv2.RANSAC，cv2.LMEDS
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1,
                                    maxIters=8000, confidence=0.999)
    elif method == "affine":  # M is 2×3的矩阵
        # 通过3点对应关系获仿射变换矩阵，use the first 3 points to do affine transform,by calling cv2.getAffineTransform()
        M = cv2.getAffineTransform(src_pts[0:3], dst_pts[0:3])
    elif method == "homo":  # 单应矩阵，M is 3×3的矩阵
        M, _ = cv2.findHomography(src_pts, dst_pts, method=0, ransacReprojThreshold=1,
                                  maxIters=8000, confidence=0.999)
        # 使用warpPerspective进行变换
    else:
        raise Exception("Error:{}".format(method))
    return M


def get_inverse_matrix(M):
    """
    计算逆变换矩阵
    --------------------------------
    Minv = get_inverse_matrix(M[:2])
    等价
    Minv = cv2.invertAffineTransform(M)
    --------------------------------
    :param M: (2×3)
    :return:
    """
    Minv = np.zeros((3, 3))
    Minv[0:2, :] = M
    Minv[2, 2] = 1
    Minv = np.linalg.pinv(Minv)
    Minv = Minv[0:2, :]
    return Minv
