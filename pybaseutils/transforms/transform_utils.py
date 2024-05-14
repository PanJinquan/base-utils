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


def get_obb_points(points, order=True):
    """
    获得旋转矩形框，即最小外接矩形的四个角点
    :param points: [shape(n,2),...,]
    :param order: 对4个点按顺时针方向进行排序:[top-left, top-right, bottom-right, bottom-left]
    :return:
    """
    return corner_utils.get_obb_points(pts=points, order=order)


def get_target_points(src_pts: np.ndarray):
    """
    根据输入的四个角点，计算其矫正后的目标四个角点,src_pts四个点分布：
        0--(w01)---1
        |          |
      (h03)      (h21)
        |          |
        3--(w23)---2
    :param src_pts:
    :return:
    """
    return corner_utils.get_target_points(src_pts=src_pts)


def get_order_points(src_pts):
    return corner_utils.get_order_points(src_pts=src_pts)


def get_image_alignment(image, src_pts, dst_pts, dsize=None, method="lstsq"):
    """
    apply affine transform实现对齐图像
    D=M*S or D=H*S
    :param image: input image
    :param src_pts: 原始点S集合(n×2)
    :param dst_pts: 目标点D集合(n×2)
    :param dsize: 变换后输出图像大小
    :param method: lstsq,estimate,affine,homo
    :return:  align_image 对齐后的图像
              M           S->D的变换矩阵(2×3)
              Minv        D->S的逆变换矩阵(2×3)
    """
    if dsize is None:
        xmax = int(max(dst_pts[:, 0]))
        ymax = int(max(dst_pts[:, 1]))
        dsize = (xmax, ymax)
    M = get_transform(src_pts, dst_pts, method=method)
    # 进行仿射变换
    if M.size == 6:  # 如果M是2×3的变换矩阵
        align_image = cv2.warpAffine(image, M=M, dsize=tuple(dsize))
        # Minv = get_transform(dst_pts, src_pts, method=method)
        Minv = get_inverse_matrix(M)
        # Minv = cv2.invertAffineTransform(M)
    else:  # 如果M是3×3的单应矩阵
        align_image = cv2.warpPerspective(image, M, dsize=tuple(dsize))
        Minv = get_transform(dst_pts, src_pts, method=method)
    return align_image, M, Minv


def image_alignment(image: np.ndarray, src_pts, dst_pts=None, dsize=(-1, -1), scale=(1.0, 1.0), method="lstsq"):
    """
    apply affine transform实现对齐图像
    :param image:
    :param src_pts:
    :param dst_pts:
    :param dsize:
    :param scale:
    :param method:
    :return:
    """
    h, w = image.shape[:2]
    if dst_pts is None:
        dst_pts = get_target_points(src_pts)
        xmin = min(dst_pts[:, 0])
        ymin = min(dst_pts[:, 1])
        xmax = max(dst_pts[:, 0])
        ymax = max(dst_pts[:, 1])
        tsize = (xmax - xmin, ymax - ymin)
    else:
        tsize = (w, h)
    if dsize is None or (dsize[0] < 0 or dsize[1] < 0):
        dsize = tsize
    # 映射居中
    tsize = np.array(tsize)
    dsize = np.array(dsize)
    dst_pts = np.array(dst_pts)
    diff = dsize - tsize
    dst_pts = dst_pts + diff / 2.
    # 缩放大小
    diff = dsize * scale - dsize
    dsize = np.array(dsize * scale, dtype=np.int32)
    dst_pts = dst_pts + diff / 2
    dst, M, Minv = get_image_alignment(image, src_pts=src_pts, dst_pts=dst_pts, dsize=tuple(dsize), method=method)
    return dst, dst_pts, M, Minv


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


def test_transform(image_file):
    from pybaseutils import image_utils
    image = cv2.imread(image_file)
    for i in range(360 * 2):
        src = image_utils.image_rotation(image.copy(), angle=i)
        mask = image_utils.get_image_mask(src)
        contours = image_utils.find_mask_contours(mask, max_nums=1)
        src_pts = image_utils.find_minAreaRect(contours, order=True)
        dst = src.copy()
        dst_pts = []
        if len(src_pts) > 0:
            src_pts = src_pts[0]
            dst, dst_pts, M, Minv = image_alignment(dst, src_pts, dsize=(-1, -1), scale=(1.2, 1.2))
        src = image_utils.draw_image_contours(src, [src_pts])
        src = image_utils.draw_landmark(src, [src_pts], color=(255, 0, 0), vis_id=True)
        dst = image_utils.draw_landmark(dst, [dst_pts], color=(0, 255, 0), vis_id=True)
        image_utils.cv_show_image("src", src, delay=10)
        image_utils.cv_show_image("dst", dst, delay=0)


if __name__ == '__main__':
    image_file = "../../data/mask/mask4.jpg"
    test_transform(image_file)
