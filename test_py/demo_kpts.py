# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-09-02 16:31:38
    @Brief  :
"""

import cv2
import numbers
import numpy as np
from pybaseutils import image_utils, file_utils

# skeleton连接线，keypoint关键点名称，num_joints关键点个数
BONES = {
    "coco_person": {
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        "keypoint": [],
        "num_joints": 17,
    },
    "hand": {
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                     [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
                     [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]],
        "keypoint": [],
        "num_joints": 21,
    }
}


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
    image = np.zeros_like(image)+255
    skeleton = BONES["coco_person"]["skeleton"]
    image = image_utils.draw_key_point_in_image(image,
                                                key_points=[kpt],
                                                pointline=skeleton,
                                                boxes=[box],
                                                thickness=2,
                                                vis_id=True)
    image_utils.cv_show_image("image", image)


if __name__ == "__main__":
    example()
