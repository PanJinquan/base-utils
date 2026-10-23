"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-10-23 15:20:57
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import copy
import numpy as np
from pybaseutils import image_utils, file_utils
from pybaseutils.transforms import transform_utils


def draw_keypoint(image, bbox, text, spts, color=(0, 255, 0), delay=0):
    kpid = list(range(len(spts)))  # 坐标点ID
    # 绘制矩形框
    if len(bbox) > 0: image = image_utils.draw_bbox_text_en(image, bbox, text=text, color=color)
    # 绘制坐标点
    if len(spts) > 0: image = image_utils.draw_points_texts(image, points=spts, texts=kpid, thickness=1, fontScale=1.0,
                                                            color=color)
    image_utils.show_image(text, image, delay=delay)
    return image


def get_meter_standard(file="img_0.png", vis=True):
    """
    获得标准的、正方形框(长宽比1:1)、无畸变、无倾斜的水表图片，作为参考点
    :return:
    """
    image = cv2.imread(file)
    h, w = image.shape[:2]
    bbox = [0, 0, w, h]  # 水表矩形框
    spts = [[65, 228], [230, 238], [145, 151]]  # 水表的三个坐标点
    npts = np.array(spts) / (w, h)  # 归一化坐标
    if vis: draw_keypoint(image, bbox, text="standard", spts=spts, color=(0, 255, 0), delay=5)
    return npts


if __name__ == "__main__":
    # TODO 获得标准水表图的参考点,npts(归一化)
    npts = get_meter_standard(file="img_0.png")
    image_file = "img_1.png"
    bbox = np.asarray([330, 180, 670, 520])  # 水表矩形框(xmin,ymin,xmax,ymax)
    spts = np.asarray([[369, 409], [569, 482], [500, 354]])  # 水表的三个坐标点
    image = cv2.imread(image_file)
    h, w = image.shape[:2]
    dsize = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    src_pts = spts
    dst_pts = npts * (w, h)
    draw_keypoint(image, bbox=bbox, text="test", spts=spts, color=(0, 255, 0), delay=10)
    dst, dst_pts, M, Minv = transform_utils.image_alignment(image,
                                                            src_pts=src_pts,
                                                            dst_pts=dst_pts,
                                                            dsize=dsize,
                                                            scale=(1.0, 1.0),
                                                            method="lstsq")
    draw_keypoint(dst, bbox=[], text="result", spts=dst_pts, color=(255, 0, 0), delay=0)
