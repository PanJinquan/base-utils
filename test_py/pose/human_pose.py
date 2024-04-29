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
from pybaseutils.pose import human_pose


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
    p = human_pose.Pose(kpt, conf, box, img=image, target="coco_person")
    # p = Pose([], [], [], img=image, target="coco_person")
    p.draw_pose(vis=True)
    # p.handR(vis=True)
    # p.handL(vis=True)
    # p.bodyUP(vis=True)
    # p.body(vis=True)
    # p.head(vis=True)
    # p.neck(vis=True)
    # p.draw_pose(vis=True)


if __name__ == "__main__":
    example()
