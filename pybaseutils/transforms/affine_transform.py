# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-05-25 09:31:17
"""
import numpy as np
import cv2
import numbers
import random
import PIL.Image as Image
import numpy as np
from typing import List, Tuple


def get_center_scale(rect, aspect_ratio, scale_rate=1.0, out_size=int or List[int, int]):
    """
    models/dataset/JointsDataset.py
    :param rect: [x, y, w, h]
    :param aspect_ratio: 长宽比=width/height
    :param scale_rate: 缩小率,值越大,相对越缩小
    :param out_size:
    :return:
    """

    def _xywh2cs(x, y, w, h, aspect_ratio, scale_rate=1.0, out_size=200):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        if isinstance(out_size, numbers.Number):
            out_size = [out_size, out_size]
        scale = np.array([w * 1.0 / out_size[0], h * 1.0 / out_size[1]], dtype=np.float32)
        # 缩小的尺度
        if center[0] != -1:
            # scale = scale * 1.25
            scale = scale * scale_rate
        return center, scale

    x, y, w, h = rect[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, scale_rate, out_size)


def affine_transform_point(point, trans):
    """
    输入原坐标点，进行仿射变换，获得变换后的坐标
    :param point: 输入坐标点 (x,y)
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return: 变换后的新坐标
    """
    new_point = np.array([point[0], point[1], 1.]).T
    new_point = np.dot(trans, new_point)  # 矩阵相乘
    return new_point[:2]


def affine_transform_points(points, trans):
    """
    输入原坐标点，进行仿射变换，获得变换后的坐标
    :param points: 输入坐标点集合，shape= (num_points,2)
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return: 变换后的新坐标
    """
    if len(points) == 0: return points
    points = np.asarray(points)
    new_point = np.ones(shape=(len(points), 3))
    new_point[:, 0:2] = points[:, 0:2]
    new_point = np.dot(trans, new_point.T).T  # 矩阵相乘
    return new_point


def affine_transform_image(image, dsize, trans, color=(0, 0, 0)):
    """
    输入原始图像，进行仿射变换，获得变换后的图像
    :param image: 输入图像
    :param dsize: 输入目标图像大小
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return:
    """
    out_image = cv2.warpAffine(image, M=trans, dsize=tuple(dsize), borderValue=color)
    return out_image


def get_kpts_affine_transform(kpts, kpts_ref, trans_type="estimate"):
    """
    估计最优的仿射变换矩阵
    :param kps: 实际关键点
    :param kpts_ref: 参考关键点
    :param trans_type:变换类型
    :return: 仿射变换矩阵
    """
    kpts = np.float32(kpts)
    kpts_ref = np.float32(kpts_ref)
    if trans_type == "estimate":
        # estimateAffine2D()可以用来估计最优的仿射变换矩阵
        trans, _ = cv2.estimateAffine2D(kpts, kpts_ref)
    elif trans_type == "affine":
        # 通过3点对应关系获仿射变换矩阵
        trans = cv2.getAffineTransform(kpts[0:3], kpts_ref[0:3])
    else:
        raise Exception("Error:{}".format(trans_type))
    return trans


def get_affine_transform(output_size,
                         center,
                         scale=[1.0, 1.0],
                         rot=0.,
                         shift=[0, 0],
                         inv=False):
    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    center = np.asarray(center)
    scale = np.asarray(scale)
    output_size = np.asarray(output_size)
    shift = np.array(shift, dtype=np.float32)
    # scale_tmp = scale * 200.0
    scale_tmp = scale * output_size
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_reference_facial_points(square=True, isshow=False):
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
    # face size[96_112] reference facial points
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    kpts_ref = np.asarray(kpts_ref)  # kpts_ref_96_112
    # for output_size=[112, 112]
    if square:
        face_size_ref = np.array(face_size_ref)
        size_diff = max(face_size_ref) - face_size_ref
        kpts_ref += size_diff / 2
        face_size_ref += size_diff

    if isshow:
        from utils import image_utils
        tmp = np.zeros(shape=(face_size_ref[1], face_size_ref[0], 3), dtype=np.uint8)
        tmp = image_utils.draw_landmark(tmp, [kpts_ref], vis_id=True)
        cv2.imshow("kpts_ref", tmp)
        cv2.waitKey(0)
    return kpts_ref


def affine_transform_for_landmarks(image, landmarks, output_size=None):
    """
    对图像和landmarks关键点进行仿生变换
    :param image:输入RGB/BGR图像
    :param landmarks:人脸关键点landmarks(5个点)
    :param output_size:输出大小
    :return:
    """
    if not output_size:
        h, w, _ = image.shape
        output_size = [w, h]
    kpts_ref = get_reference_facial_points(square=True, isshow=False)
    alig_faces = []
    warped_landmarks = []
    for landmark in landmarks:
        trans = get_kpts_affine_transform(kpts=landmark, kpts_ref=kpts_ref, trans_type="estimate")
        trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
        alig_faces.append(trans_image)
        landmark = affine_transform_points(landmark, trans)
        warped_landmarks.append(landmark)
    return alig_faces, warped_landmarks


def rotate_points(points, centers, angle, height):
    """
    eg.:
    height, weight, d = image.shape
    point1 = [[300, 200],[50, 200]]
    point1 = np.asarray(point1)
    center = [[200, 200]]
    point3 = rotate_points(point1, center, angle=30, height=height)
    :param points:
    :param centers:
    :param angle:
    :param height:
    :return:
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if not isinstance(centers, np.ndarray):
        centers = np.asarray(centers)
    dst_points = points.copy()
    # 将图像坐标转换到平面坐标
    dst_points[:, 1] = height - dst_points[:, 1]
    centers[:, 1] = height - centers[:, 1]
    x = (dst_points[:, 0] - centers[:, 0]) * np.cos(np.pi / 180.0 * angle) - (
            dst_points[:, 1] - centers[:, 1]) * np.sin(np.pi / 180.0 * angle) + centers[:, 0]
    y = (dst_points[:, 0] - centers[:, 0]) * np.sin(np.pi / 180.0 * angle) + (
            dst_points[:, 1] - centers[:, 1]) * np.cos(np.pi / 180.0 * angle) + centers[:, 1]
    # 将平面坐标转换到图像坐标
    y = height - y
    dst_points[:, 0] = x
    dst_points[:, 1] = y
    return dst_points


def get_boxes2points(boxes):
    """
    :param boxes:
    :return:
    """
    # (num_boxes,4)=(num_boxes,xmin,ymin,xmax)
    num_boxes = len(boxes)
    xmin = boxes[:, 0:1]
    ymin = boxes[:, 1:2]
    xmax = boxes[:, 2:3]
    ymax = boxes[:, 3:4]
    t1 = np.hstack([xmin, ymin])
    t2 = np.hstack([xmin, ymax])
    t3 = np.hstack([xmax, ymin])
    t4 = np.hstack([xmax, ymax])
    # (num_boxes,8)=(num_boxes,xmin,ymin,xmax,ymax,xmin,ymax,xmax,ymin)
    points = np.hstack([t1, t4, t2, t3])
    # dst_boxes = dst_boxes[:, 0:4]
    points = points.reshape(num_boxes, -1, 2)  # (num_boxes,box_point(4),2)
    return points, num_boxes


def get_points2bboxes(points):
    """
    :param boxes:
    :return:
    """
    xmin = np.min(points[:, :, 0:1], axis=1)
    ymin = np.min(points[:, :, 1:2], axis=1)
    xmax = np.max(points[:, :, 0:1], axis=1)
    ymax = np.max(points[:, :, 1:2], axis=1)
    t1 = np.hstack([xmin, ymin])
    t2 = np.hstack([xmax, ymax])
    boxes = np.hstack([t1, t2])
    return boxes


def affine_transform_for_boxes(image, boxes, output_size=None, rot=0, inv=False):
    """
    对图像和boxes进行仿生变换
    :param image:输入RGB/BGR图像
    :param boxes:检测框
    :param output_size:输出大小
    :param rot:旋转角度，PS：旋转时，由于boxes只包含左上角和右下角的点，
               所以旋转时box的矩形框会变得比较扁
    :return:
    """
    boxes = np.asarray(boxes)
    h, w, _ = image.shape
    center = (int(w / 2), int(h / 2))
    scale = [1.0, 1.0]
    if not output_size:
        output_size = [w, h]
    trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
    trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
    points, num_boxes = get_boxes2points(boxes)
    for i in range(num_boxes):
        points[i, :] = affine_transform_points(points[i, :], trans)
    boxes = get_points2bboxes(points)
    return trans_image, boxes


class RandomAffineTransform(object):
    def __init__(self, out_size, scale_rate=1.0, degrees=5, rot=0.5, flip=True, flip_index=[], color=(0, 0, 0)):
        """
        :param out_size: 输出图像分辨率大小
        :param aspect_ratio: 长宽比=width/height
        :param scale_rate: 缩小率,值越大,相对越缩小
        :param degrees:随机旋转的角度
        :param rot:随机旋转的概率
        :param flip:是否随机左右翻转(概率p=0.5)
        :param flip_index: 翻转后，对于的关键点也进行翻转，如果输入的关键点没有左右关系的，
                           请设置flip_index，以便翻转时，保证index的关系
                           如，对于人脸关键点：flip_index=[2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.out_size = out_size
        self.rot = rot
        self.flip = flip
        self.flip_index = flip_index
        self.scale_rate = scale_rate
        self.aspect_ratio = self.out_size[0] * 1.0 / self.out_size[1]  # 192/256=0.75
        self.center = None
        self.scale = None
        self.color = color

    def set_center_scale(self, center, scale):
        self.center = center
        self.scale = scale

    def __call__(self, image, points):
        is_pil = isinstance(image, Image.Image)
        image = np.asarray(image) if is_pil else image
        height, width = image.shape[:2]
        angle = 0
        if random.random() < self.rot:
            angle = random.uniform(self.degrees[0], self.degrees[1])
        center, scale = self.center, self.scale
        # print(angle)
        if not center or not scale:
            rect = [0, 0, width, height]  # x, y, w, h
            center, scale = get_center_scale(rect, self.aspect_ratio, self.scale_rate, out_size=self.out_size)
        if self.flip and random.random() <= 0.5:
            if len(points) > 0:
                points[:, 0] = width - points[:, 0]
                if self.flip_index: points = points[self.flip_index]
            image = image[:, ::-1, :]
            center[0] = width - center[0]
        image, points, trans = AffineTransform.affine_transform_for_image_points(image, points, self.out_size,
                                                                                 center,
                                                                                 scale,
                                                                                 rot=angle,
                                                                                 inv=False,
                                                                                 color=self.color)
        image = Image.fromarray(image) if is_pil else image
        return image, points


class AffineTransform(object):
    @staticmethod
    def affine_transform_for_boxes(boxes, output_size, center, scale, rot=0, inv=False, **kwargs):
        """
        根据center, scale对bbox，或者kwargs进行仿生变换
        :param boxes: shape(num_boxes,(xmin,ymin,xmax,ymax))
        :param output_size:
        :param center: 旋转中心点
        :param scale: 缩放因子
        :param rot: 旋转角度
        :param inv: True: 仿生变换,False:反变换
        :param kwargs: {"key":shape(num_boxes,x1,y1,x2,y2,...,xn,yn)}
        :return:
        """
        trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
        points, nums = get_boxes2points(boxes)  # points is (num_boxes,box_point(4),2)
        for i in range(nums):
            points[i, :] = affine_transform_points(points[i, :], trans)
        boxes = get_points2bboxes(points)
        if kwargs:
            for k in kwargs.keys():
                points = np.asarray(kwargs[k]).reshape(-1, 2)
                points = affine_transform_points(points, trans)
                kwargs[k] = points.reshape(nums, -1)
        return boxes, trans, kwargs

    @staticmethod
    def affine_transform_for_points(points, output_size, center, scale, rot=0, inv=False):
        # points is (num-points,2)
        trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
        points = affine_transform_points(points, trans)
        return points, trans

    @staticmethod
    def affine_transform_for_image_points(image, points, output_size,
                                          center, scale=[1.0, 1.0], rot=0.,
                                          inv=False, color=(0, 0, 0)):
        """
        h, w = image.shape[:2]
        rect = [0, 0, w, h]
        scale_rate = 1.0
        output_size = [320, 320]
        aspect_ratio = output_size[0] * 1.0 / output_size[1]  # 192/256=0.75
        center, scale = get_center_scale(rect, aspect_ratio, scale_rate, out_size=output_size)
        :param image:
        :param points:(N,2)
        :param output_size:(w,h)
        :param center: 中心点，通过get_center_scale获得
        :param scale:  缩放大小，通过get_center_scale获得
        :param rot:
        :param inv:
        :param color:
        :return:
        """
        trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
        points = affine_transform_points(points, trans)
        if inv:
            dsize = (int(center[0] * 2), int(center[1] * 2))
        else:
            dsize = tuple(output_size)
        image = cv2.warpAffine(image, trans, dsize=dsize, flags=cv2.INTER_LINEAR, borderValue=color)
        return image, points, trans

    @staticmethod
    def affine_transform(image, boxes, output_size, rot=0, **kwargs):
        """
        对图像和boxes进行仿生变换
        :param image:
        :param boxes:
        :param output_size:
        :param rot: 旋转角度
        :return:
        """
        boxes = np.asarray(boxes)
        h, w, _ = image.shape
        center = (int(w / 2), int(h / 2))
        long_side = max([w / output_size[0], h / output_size[1]])
        scale = [long_side, long_side]
        boxes, trans, kwargs = AffineTransform.affine_transform_for_boxes(boxes,
                                                                          output_size,
                                                                          center,
                                                                          scale,
                                                                          rot=rot,
                                                                          inv=False,
                                                                          **kwargs)
        trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
        return trans_image, boxes, center, scale, kwargs

    @staticmethod
    def inverse_affine_transform(boxes, output_size, center, scale, rot=0, **kwargs):
        """对图像和boxes进行反变换"""
        boxes, trans, kwargs = AffineTransform.affine_transform_for_boxes(boxes,
                                                                          output_size,
                                                                          center,
                                                                          scale,
                                                                          rot=rot,
                                                                          inv=True,
                                                                          **kwargs)
        return boxes, kwargs


def demo_for_landmarks():
    image_path = "face.jpg"
    image = image_utils.read_image(image_path)
    # face detection from MTCNN
    bbox_score = np.asarray([[69.48486808, 58.12609892, 173.92575279, 201.95947894, 0.99979943]])
    landmarks = np.asarray([[[103.97721, 119.6718],
                             [152.35837, 113.06249],
                             [136.67535, 142.62952],
                             [112.62607, 171.1305],
                             [154.60092, 165.12515]]])
    bboxes = bbox_score[:, :4]
    scores = bbox_score[:, 4:]
    image = image_utils.draw_landmark(image, landmarks)
    image_utils.cv_show_image("image", image, use_rgb=False)
    alig_faces, warped_landmarks = affine_transform_for_landmarks(image, landmarks, output_size=[256, 256])
    for i in range(len(alig_faces)):
        alig_face = image_utils.draw_landmark(alig_faces[i], [warped_landmarks[i]], color=(0, 255, 0))
        image_utils.cv_show_image("image", alig_face, use_rgb=False)


def demo_for_image_boxes():
    image_path = "test.jpg"
    bboxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    output_size = [320, 320]
    image = image_utils.read_image(image_path)
    image_utils.show_image_boxes("src", image, bboxes, waitKey=10)
    for i in range(360):
        trans_image, trans_boxes = affine_transform_for_boxes(image, bboxes, output_size=output_size, rot=i)
        print("shape:{},bboxes     ：{}".format(image.shape, bboxes))
        print("shape:{},trans_boxes：{}".format(trans_image.shape, trans_boxes))
        image_utils.show_image_boxes("trans", trans_image, trans_boxes, color=(0, 255, 0))


def demo_for_image_affine_transform():
    image_path = "test.jpg"
    bboxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    land_mark = [[[122.44442, 54.193676],
                  [147.6293, 56.77364],
                  [135.35794, 74.66961],
                  [120.94379, 83.858765],
                  [143.35617, 86.417175]],
                 [[258.14902, 287.81662],
                  [281.83157, 281.46664],
                  [268.39877, 306.3493],
                  [265.5242, 318.80936],
                  [286.5602, 313.99652]]]
    land_mark = np.asarray(land_mark).reshape(-1, 10)
    output_size = [320, 320]
    image = image_utils.read_image(image_path)
    image_utils.show_image_boxes("src", image, bboxes, delay=10)
    at = AffineTransform()
    for i in range(360):
        trans_image, trans_boxes, center, scale, kwargs = at.affine_transform(image,
                                                                              bboxes,
                                                                              output_size=output_size,
                                                                              rot=i,
                                                                              land_mark=land_mark)
        img = image.copy()
        image_boxes, kwargs = at.inverse_affine_transform(trans_boxes, output_size, center, scale, rot=i, **kwargs)
        points = kwargs["land_mark"].reshape(len(trans_boxes), -1, 2)
        print("shape:{},bboxes     ：{}".format(image.shape, bboxes))
        print("shape:{},trans_boxes：{}".format(trans_image.shape, trans_boxes))
        image_utils.show_image_boxes("trans", trans_image, trans_boxes, color=(0, 255, 0), delay=1)
        img = image_utils.draw_landmark(img, points, color=(0, 255, 0))
        image_utils.show_image_boxes("img", img, image_boxes, color=(0, 255, 0))


def demo_for_affine_transform_for_image_points():
    image_path = "test.jpg"
    points = [[[122.44442, 54.193676],
               [147.6293, 56.77364],
               [135.35794, 74.66961],
               [120.94379, 83.858765],
               [143.35617, 86.417175]],
              [[258.14902, 287.81662],
               [281.83157, 281.46664],
               [268.39877, 306.3493],
               [265.5242, 318.80936],
               [286.5602, 313.99652]]]
    points = np.asarray(points).reshape(-1, 2)
    at = AffineTransform()
    image = image_utils.read_image(image_path)
    h, w = image.shape[:2]
    rect = [0, 0, w, h]
    scale_rate = 1.0
    output_size = [320, 320]
    aspect_ratio = output_size[0] * 1.0 / output_size[1]  # 192/256=0.75
    center, scale = get_center_scale(rect, aspect_ratio, scale_rate, out_size=output_size)
    timage, tpoints, trans = at.affine_transform_for_image_points(image, points, output_size, center, scale=scale)
    rimage, rpoints, trans = at.affine_transform_for_image_points(timage, tpoints, output_size, center, scale=scale,
                                                                  inv=True)
    image = image_utils.draw_landmark(image, [points], color=(255, 0, 0))
    image_utils.cv_show_image("image", image, delay=1)
    timage = image_utils.draw_landmark(timage, [tpoints], color=(0, 255, 0))
    image_utils.cv_show_image("timage", timage, delay=1)
    rimage = image_utils.draw_landmark(rimage, [rpoints], color=(0, 0, 255))
    image_utils.cv_show_image("rimage", rimage)


if __name__ == "__main__":
    from pybaseutils import image_utils

    # demo_for_landmarks()
    # demo_for_image_boxes()
    # demo_for_image_affine_transform()
    demo_for_affine_transform_for_image_points()
