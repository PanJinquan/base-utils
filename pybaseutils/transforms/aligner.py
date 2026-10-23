"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-10-23 15:20:57
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np
from pybaseutils import image_utils, file_utils


def transform_boxes(src_boxes, M, inv=False):
    """
    将矩形框通过变换矩阵M映射到新图，返回映射后的外接矩形框
    :param src_boxes: (N,4) 矩形框(xmin,ymin,xmax,ymax)
    :param M: 变换矩阵，2×3仿射矩阵或3×3单应矩阵
    :param inv: 是否逆变换，默认False
           注意：逆变换回来的 src_box 仍是外接框（比原始框大），因为正向 dst_box 已丢失旋转信息；如要精确框，需保留变换后的 4 个角点。
    :return: dst_boxes (N,4) 变换后的外接矩形框(xmin,ymin,xmax,ymax)
    """
    src_boxes = np.asarray(src_boxes, dtype=np.float32).reshape(-1, 4)
    xmin, ymin, xmax, ymax = src_boxes.T
    # 每个框的4个角点(左上,右上,右下,左下)，shape=(N,4,2)
    corners_x = np.stack([xmin, xmax, xmax, xmin], axis=1)
    corners_y = np.stack([ymin, ymin, ymax, ymax], axis=1)
    corners = np.stack([corners_x, corners_y], axis=-1)
    # 一次性变换所有角点，再还原为(N,4,2)
    corners = transform_points(corners.reshape(-1, 2), M, inv=inv).reshape(-1, 4, 2)
    # 取4个角点的外接矩形
    dst_boxes = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
    return dst_boxes


def transform_points(src_pts, M, inv=False):
    """
    将坐标点通过变换矩阵M映射到新图，返回映射后的坐标点
    :param src_pts: (N,2)
    :param M: 变换矩阵，2×3仿射矩阵或3×3单应矩阵
    :param inv: 是否逆变换，默认False
    :return: dst_pts (N,2) 变换后的坐标点
    """
    src_pts = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
    if inv: M = get_inverse_matrix(M)
    if M.shape[0] == 2:  # 2×3仿射矩阵
        dst_pts = cv2.transform(src_pts[None], M)[0]
    else:  # 3×3单应矩阵，perspectiveTransform会自动做透视除法
        dst_pts = cv2.perspectiveTransform(src_pts[None], M)[0]
    return dst_pts


def transform_image(src_img, M, dsize, inv=False):
    """
    将图像通过变换矩阵M映射到新图，返回变换后的图像
    :param src_img: 原图图像
    :param M: 变换矩阵，2×3仿射矩阵或3×3单应矩阵
    :param dsize: 输出图像大小(w,h)
    :param inv: 是否逆变换，默认False
    :return: dst_img 变换后的图像
    """
    if inv: M = get_inverse_matrix(M)
    if M.size == 6:  # 如果M是2×3的变换矩阵
        dst_img = cv2.warpAffine(src_img, M, dsize=dsize, flags=cv2.INTER_LINEAR)
    else:  # 如果M是3×3的单应矩阵
        dst_img = cv2.warpPerspective(src_img, M, dsize=tuple(dsize))
    return dst_img


def align_bbox_points(src_box, src_pts, ref_pts, ref_size: tuple, method="affine"):
    """
    目标矫正算法: 对矩形框和点进行矫正
    :param src_box: shape=(4,),目标在原图的矩形框(xmin,ymin,xmax,ymax)
    :param src_pts: shape=(N,2)目标在原图的关键点,N≥3
    :param ref_pts: shape=(N,2)目标参考图的关键点,src_pts与ref_pts必须一一对应
    :param ref_size: tuple(w,h)目标参考图的大小
    :param method: 变换方法，默认affine,可选值：lstsq,estimate,affine,homo
    :return: 返回矫正后矩形框dst_box, 关键点dst_pts, dst_size,变换矩阵M
    """
    src_pts = np.asarray(src_pts, dtype=np.float32)
    ref_pts = np.asarray(ref_pts, dtype=np.float32)
    xmin, ymin, xmax, ymax = src_box
    # 保持标准目标图的长宽比：取检测框与标准图在两个方向上的最大缩放比例作为统一缩放系数
    scale = max((xmax - xmin) / ref_size[0], (ymax - ymin) / ref_size[1])
    # 矫正图大小 = 标准图大小 × scale，长宽比与标准图保持一致
    dst_size = (int(round(ref_size[0] * scale)), int(round(ref_size[1] * scale)))  # (w,h)
    # 标准关键点 -> 矫正图上的目标坐标
    ref_pts = ref_pts * scale
    # 3组一一对应的点即可确定仿射变换(透视变换在只有3个点时退化为仿射变换)
    M = get_transform(src_pts, ref_pts, method=method)
    # 将原图矩形框的4个角点映射到矫正图，取外接矩形作为矫正后的目标框
    dst_box = transform_boxes([src_box], M)[0]
    # 关键点映射结果，与标准关键点位置一致
    dst_pts = transform_points(src_pts, M)
    return dst_box, dst_pts, dst_size, M


def align_image_bbox_points(src_img, src_box, src_pts, ref_pts, ref_size: tuple, method="affine"):
    """
    目标矫正算法: 对图像、矩形框和点进行矫正
    :param src_img: shape=(h,w,3) 原图，np.ndarray
    :param src_box: shape=(4,),目标在原图的矩形框(xmin,ymin,xmax,ymax)
    :param src_pts: shape=(N,2)目标在原图的关键点,N≥3
    :param ref_pts: shape=(N,2)目标参考图的关键点,src_pts与ref_pts必须一一对应
    :param ref_size: tuple(w,h)目标参考图的大小
    :param method: 变换方法，默认affine,可选值：lstsq,estimate,affine,homo
    :return: 返回矫正后图片dst_img, 矩形框dst_box, 关键点dst_pts, 变换矩阵M
    """
    dst_box, dst_pts, dst_size, M = align_bbox_points(src_box, src_pts, ref_pts, ref_size=ref_size, method=method)
    # 整图映射到与标准图同比例的画布，得到与标准图布局一致、无倾斜的矫正图
    dst_img = transform_image(src_img, M, dsize=dst_size)
    return dst_img, dst_box, dst_pts, dst_size, M


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
        M = np.float32([[H[0, 0], H[1, 0], H[2, 0]], [H[0, 1], H[1, 1], H[2, 1]]])
    elif rank == 2:
        M = np.float32([[H[0, 0], H[1, 0], 0], [H[0, 1], H[1, 1], 0]])
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
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1, maxIters=8000,
                                    confidence=0.999)
    elif method == "affine":  # M is 2×3的矩阵
        # 通过3点对应关系获仿射变换矩阵，use the first 3 points to do affine transform,by calling cv2.getAffineTransform()
        M = cv2.getAffineTransform(src_pts[0:3], dst_pts[0:3])
    elif method == "homo":  # 单应矩阵，M is 3×3的矩阵
        M, _ = cv2.findHomography(src_pts, dst_pts, method=0, ransacReprojThreshold=1, maxIters=8000, confidence=0.999)
        # 使用warpPerspective进行变换
    else:
        raise Exception("Error:{}".format(method))
    return M


def get_inverse_matrix(M):
    """
    计算变换矩阵的逆矩阵
    :param M: 2×3仿射矩阵或3×3单应矩阵
    :return: 逆变换矩阵
    """
    if M.shape[0] == 2:  # 2×3仿射矩阵
        return cv2.invertAffineTransform(M)
    else:  # 3×3单应矩阵
        return np.linalg.inv(M)


def draw_keypoint(image, bbox, text, pts, color=(0, 255, 0), thick=1, delay=0):
    """
    绘图
    :param image:
    :param bbox:
    :param text:
    :param pts:
    :param color:
    :param delay:
    :return:
    """
    bbox = np.asarray(bbox, dtype=np.int32)
    pts = np.asarray(pts, dtype=np.int32)
    kpid = list(range(len(pts)))  # 坐标点ID
    # 绘制矩形框
    if len(bbox) > 0:
        image = image_utils.draw_bbox_text_en(image, bbox, text=text, color=color)
    # 绘制坐标点
    if len(pts) > 0:
        image = image_utils.draw_points_texts(image, points=pts, texts=kpid, thickness=thick, fontScale=1.0,
                                              color=color)
    # cv2.imwrite("{}.jpg".format(text), image)
    image_utils.show_image(text, image, delay=delay)
    return image


def get_ref_object(file="img_0.png", vis=True):
    """
    获得标准的、无畸变、无倾斜的目标图片，作为参考点
    :return:
    """
    image = cv2.imread(file)
    h, w = image.shape[:2]
    bbox = [0, 0, w, h]  # 目标矩形框
    ref_pts = [[65, 228], [230, 238], [145, 151]]  # 目标的三个坐标点
    ref_size = (w, h)
    if vis: draw_keypoint(image, bbox, text="standard", pts=ref_pts, color=(0, 255, 0), delay=5)
    return ref_pts, ref_size


if __name__ == "__main__":
    """
    矫正算法：已知标准目标图的三个点npts，以及测试图的目标矩形框src_box和三个点src_pts
    npts和src_pts一一对应，利用Opencv 透视变换等方法实现目标矫正。
    """
    # TODO 获得标准目标图的参考点,npts(归一化)
    ref_pts, ref_size = get_ref_object(file="img_0.png")
    src_img = cv2.imread("img_1.png")  # 测试图
    src_box = np.asarray([330, 180, 670, 520])  # 测试图目标矩形框(xmin,ymin,xmax,ymax)
    src_pts = np.asarray([[369, 409], [569, 482], [500, 354]])  # 测试图目标的三个坐标点，与npts一一对应
    draw_keypoint(src_img, bbox=src_box, text="test", pts=src_pts, color=(0, 255, 0), thick=3, delay=10)
    dst_img, dst_box, dst_pts, dst_size, M = align_image_bbox_points(src_img, src_box=src_box, src_pts=src_pts,
                                                                     ref_pts=ref_pts, ref_size=ref_size)
    draw_keypoint(dst_img, bbox=dst_box, text="result", pts=dst_pts, color=(255, 0, 0), thick=2, delay=10)
    # 逆变换样例：已知dst_box,dst_pts和M，反算src_box,src_pts（关键点可精确还原，矩形框为外接框）
    src_box_inv = transform_boxes([dst_box], M, inv=True)[0]
    src_pts_inv = transform_points(dst_pts, M, inv=True)
    draw_keypoint(src_img, bbox=src_box_inv, text="inverse", pts=src_pts_inv, color=(0, 0, 255), thick=2, delay=0)
