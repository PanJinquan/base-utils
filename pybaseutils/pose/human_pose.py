# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-04 09:37:34
    @Brief  :
"""
import cv2
import numbers
import numpy as np
from pybaseutils import image_utils, file_utils
from pybaseutils.pose import pose_utils as base_utils
from pybaseutils.pose import bones_utils




class Pose(object):
    def __init__(self, kpt, conf, box=[], score=1.0, img=None, threshold=0.3, target="coco_person"):
        """
        :param kpt: 关键点坐标,shape is (17,2)
        :param conf: 关键点置信度,shape is (17,)
        :param box: 人体框(xmin,ymin,xmax,ymax)
        :param score: 人体框置信度
        :param img: 图片
        :param threshold: 关键点置信度阈值，如果conf平均值小于该值，则check=False
        :param target: 默认目标为coco_person
        """
        self.kpt = np.asarray(kpt) if isinstance(kpt, list) else kpt
        self.conf = np.asarray(conf) if isinstance(conf, list) else conf
        self.box = box  # 人体框(xmin,ymin,xmax,ymax)
        self.score = score  # 人体框置信度
        self.img = img
        self.skeleton = bones_utils.BONES[target]['skeleton']
        self.colors =bones_utils.BONES[target]['colors']
        self.num_joints =bones_utils.BONES[target]['num_joints']
        self.check = len(self.kpt) == self.num_joints
        self.front = self.check_face_front()  # 正向/背向镜头
        if self.check:
            index = [5, 6, 12, 11]
            self.check = self.__check_pose(self.kpt[index], self.conf[index], th=threshold)
        self.baseline = self.__baseline()

    def __baseline(self):
        """计算身体的baseline，该长度作为人体的相对长度"""
        if not self.check: return None
        dl = base_utils.distance(p1=self.kpt[5], p2=self.kpt[11])
        dr = base_utils.distance(p1=self.kpt[6], p2=self.kpt[12])
        d = (dl + dr) / 2
        return d

    def check_face_front(self, ):
        """
        正向/背向镜头
        :return:
        """
        r = False
        if self.check:
            indexL = [6, 8, 10]
            indexR = [5, 7, 9]
            kptL = np.mean(self.kpt[indexL], axis=0)
            kptR = np.mean(self.kpt[indexR], axis=0)
            r = kptR[0] > kptL[0]  # 如果右边X坐标大于左边坐标，则是正向
        return r

    def neck(self, th=0.2, vis=False):
        """
        获得脖子关键点和box
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        # 肩膀连线中心点作为脖子位置
        if self.check:
            index = [5, 6]
            kpt = [np.mean(self.kpt[index], axis=0)]
            conf = [np.mean(self.conf[index], axis=0)]
            r = (self.baseline * 0.2, self.baseline * 0.25)
            box = base_utils.create_box_from_point(c=kpt[0], r=r)
            c = self.__check_pose(kpt, conf, th=th)
        else:
            kpt, box, conf, c = [], [], [], False
        info = {"kpt": kpt, "box": box, "conf": conf, "check": c, "name": "neck"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    def body(self, th=0.2, vis=False):
        """
        获得全身关键点和box
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        if self.check:
            box = base_utils.points2box(self.kpt)
            c = self.__check_pose(self.kpt, self.conf, th=th)
        else:
            box, c = [], False
        info = {"kpt": self.kpt, "box": box, "conf": self.conf, "check": c, "name": "body"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    def bodyUP(self, square=False, th=0.2, vis=False):
        """
        获得上半身关键点和box
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        if self.check:
            index = [5, 6, 12, 11]
            kpt = self.kpt[index]
            conf = self.conf[index]
            box = base_utils.points2box(kpt)
            c = self.__check_pose(kpt, conf, th=th)
            if square: box = image_utils.get_square_bboxes([box], use_max=True)[0]
        else:
            kpt, box, conf, c = [], [], [], False
        info = {"kpt": kpt, "box": box, "conf": conf, "check": c, "name": "bodyUP"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    def hand_touch(self, box: list = [], th=0.2, vis=False):
        """
        手部与目标box有接触
        :param box:目标框(xmin,ymin,xmax,ymax)
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        handL = self.handL(th=th, vis=False)
        handR = self.handR(th=th, vis=False)
        if self.check:
            handL["iou"] = base_utils.cal_iou(box, handL.get("box", []))
            handR["iou"] = base_utils.cal_iou(box, handR.get("box", []))
        else:
            handL["iou"] = -1
            handR["iou"] = -1
        if vis:
            self.draw_info(info=handL, color=(255, 0, 0), vis=False, delay=1)
            self.draw_info(info=handR, color=(255, 0, 0), vis=False, delay=1)
            self.draw_info(info={"box": box, "name": "target"}, color=(0, 255, 0), vis=True, delay=0)
        return handL, handR

    def handL(self, th=0.2, vis=False):
        """
        获得左手关键点和box
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        if self.check:
            index = [8, 10]
            kpt = self.kpt[index]
            conf = self.conf[index]
            r = self.baseline * 0.2
            center = base_utils.extend_line(p1=self.kpt[8], p2=self.kpt[10], scale=(0.3, 0.3))
            box = base_utils.create_box_from_point(c=center, r=r)
            c = self.__check_pose(kpt, conf, th=th)
        else:
            kpt, box, conf, c = [], [], [], False
        info = {"kpt": kpt, "box": box, "conf": conf, "check": c, "name": "handL"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    def handR(self, th=0.2, vis=False):
        """
        获得右手关键点和box
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        if self.check:
            index = [7, 9]
            kpt = self.kpt[index]
            conf = self.conf[index]
            r = self.baseline * 0.2
            center = base_utils.extend_line(p1=self.kpt[7], p2=self.kpt[9], scale=(0.3, 0.3))
            box = base_utils.create_box_from_point(c=center, r=r)
            c = self.__check_pose(kpt, conf, th=th)
        else:
            kpt, box, conf, c = [], [], [], False
        info = {"kpt": kpt, "box": box, "conf": conf, "check": c, "name": "handR"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    def head(self, square=True, th=0.2, vis=False):
        """
        获得头部关键点和box
        :param square:
        :param th: 置信度阈值
        :param vis:
        :return:
        """
        if self.check:
            index = [0, 1, 2, 3, 4]
            kpt = self.kpt[index]
            conf = self.conf[index]
            box = base_utils.points2box(kpt)
            c = self.__check_pose(kpt, conf, th=th)
            if square: box = image_utils.get_square_bboxes([box], use_max=True)[0]
        else:
            kpt, box, conf, c = [], [], [], False
        info = {"kpt": kpt, "box": box, "conf": conf, "check": c, "name": "head"}
        if vis: self.draw_info(info=info, vis=vis)
        return info

    @staticmethod
    def __check_pose(kpt, conf, th=0.2):
        """
        检测关键点的是否合法
        :param kpt:人体关键点
        :param conf:人体关键点置信度
        :param th:置信度阈值
        :return: True 合法；False 不合法
        """
        if isinstance(kpt, list): kpt = np.asarray(kpt)
        if isinstance(conf, list): conf = np.asarray(conf)
        if len(kpt) == 0: return False
        p = np.prod(kpt, axis=1)
        r = sum(p > 0) > 0 and np.mean(conf) > th  # 置信度要超过一定阈值
        return r

    def draw_pose(self, vis=False):
        """
        :param skeleton:  关键点连接顺序
        :return:
        """
        image = image_utils.draw_key_point_in_image(self.img,
                                                    key_points=[self.kpt],
                                                    pointline=self.skeleton,
                                                    boxes=[self.box],
                                                    thickness=2,
                                                    colors=self.colors,
                                                    vis_id=True)
        if vis: image_utils.cv_show_image("image", image)
        return image

    def draw_info(self, info: dict, color=(255, 0, 0), vis=False, delay=0):
        """
        :param info:
        :param vis:
        :return:
        """
        # kpt = info.get("kpt", [])
        box = [info.get("box", [])]
        name = [info.get("name", [])]
        boxes = [b for b in box if b]
        texts = [n for n in name if n]
        image = image_utils.draw_image_bboxes_text(self.img,
                                                   boxes=boxes,
                                                   boxes_name=texts,
                                                   thickness=2,
                                                   fontScale=1.0,
                                                   color=color,
                                                   drawType="simple")
        if vis: image_utils.cv_show_image(name[0], image, delay=delay)
        return image


def example():
    kpts = [[405.20834, 71.875, 0.9536487],
            [415.625, 61.458332, 0.96337986],
            [384.375, 61.458332, 0.9244619],
            [436.45834, 61.458332, 0.99999976],
            [363.54166, 71.875, 0.98517716],
            [488.54166, 165.625, 0.9580497],
            [321.875, 165.625, 0.87711847],
            [551.0417, 248.95833, 0.88645184],
            [248.95833, 259.375, 0.8369312],
            [561.4583, 342.70834, 0.8438214],
            [238.54167, 342.70834, 0.8260272],
            [457.29166, 394.79166, 0.8504467],
            [363.54166, 394.79166, 0.85016763],
            [467.70834, 571.875, 0.9571377],
            [363.54166, 561.4583, 0.929651],
            [467.70834, 738.5417, 0.9448692],
            [363.54166, 738.5417, 0.9097275]]
    kpts = np.asarray(kpts)
    conf = kpts[:, 2]
    kpt = kpts[:, 0:2]
    box = [0, 0, 800, 800]
    image_file = "pose.jpg"
    image = cv2.imread(image_file)
    p = Pose(kpt, conf, box, img=image, target="coco_person")
    # p = Pose([], [], [], img=image, target="coco_person")
    p.draw_pose(vis=True)
    p.handR(vis=True)
    p.handL(vis=True)
    # p.bodyUP(vis=True)
    # p.body(vis=True)
    # p.head(vis=True)
    # p.neck(vis=True)
    # p.draw_pose(vis=True)


if __name__ == "__main__":
    example()
