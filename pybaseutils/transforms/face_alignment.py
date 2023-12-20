# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-12-14 15:09:34
    @Brief  :
"""
import cv2
import numpy as np


def get_reference_facial_points(out_size=(112, 112), square=True, vis=False):
    """
    获得人脸参考关键点,目前支持两种输入的参考关键点,即[96, 112]和[112, 112]
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    ==================
    face_size_ref = [112, 112]
    kpts_ref = [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]

    ==================
    square = True, crop_size = (112, 112)
    square = False,crop_size = (96, 112),
    :param square: True is [112, 112] or False is [96, 112]
    :param isshow: True or False,是否显示
    :return:
    """
    size_ref = (96, 112)
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    dst_pts = np.asarray(kpts_ref, dtype=np.float32)
    if square or out_size[0] != size_ref[0] or out_size[1] != size_ref[1]:
        size_ref = np.array(size_ref, dtype=np.float32)
        maxL = max(size_ref)
        wh_diff = maxL - size_ref
        dst_pts = dst_pts + wh_diff / 2.0
        dst_pts = dst_pts * out_size / maxL
        size_ref = size_ref + wh_diff
        size_ref = size_ref * out_size / maxL
    if vis:
        from pybaseutils import image_utils
        tmp = np.zeros(shape=(int(size_ref[1]), int(size_ref[0]), 3), dtype=np.uint8)
        tmp = image_utils.draw_landmark(tmp, [dst_pts], vis_id=True)
        cv2.imshow("kpts_ref", tmp)
        cv2.waitKey(0)
    return dst_pts


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


def alignment_image(image, src_pts, dst_pts, out_size, method="lstsq"):
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


if __name__ == "__main__":
    from pybaseutils import image_utils

    out_size = (224, 224)
    dst_pts = get_reference_facial_points(out_size=out_size, vis=False)
    src_pts = [[305.6163, 254.10919],
               [388.11459, 268.15384],
               [352.84979, 324.34232],
               [287.70984, 342.08038],
               [370.14178, 354.08649]]
    src_pts = np.asarray(src_pts, dtype=np.float32)
    M = get_transform(src_pts, dst_pts, method="lstsq")
    # M = get_transform(src_pts, dst_pts, method="estimate")
    print(np.asarray(M, dtype=np.float32))
    image_file = "../data/test02.jpg"
    image = cv2.imread(image_file)
    align_image, M, Minv = alignment_image(image, src_pts, dst_pts, out_size, method="lstsq")
    print("M   :\n", M)
    print("Minv:\n", Minv)
    image_utils.cv_show_image("image", image, delay=10)
    image_utils.cv_show_image("align_image", align_image)
