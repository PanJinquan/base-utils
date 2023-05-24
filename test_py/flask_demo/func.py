# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-16 08:53:32
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import image_utils


# img = ins_get_image('t1')

def getRet(filename):
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    cv2.imwrite("./static/ldh_output.jpg", img)
    image_utils.cv_show_image("image", img)
