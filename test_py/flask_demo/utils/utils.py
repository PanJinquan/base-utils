# -*- coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2023-05-16 08:53:11
    @Brief  : https://www.runoob.com/flask/flask-tutorial.html
"""
import io
import os.path

import cv2
import numpy as np
from typing import List, Tuple
from werkzeug.datastructures import FileStorage
from pybaseutils import file_utils


def image2bytes(image):
    _, image = cv2.imencode('.jpg', image)
    return image.tobytes()


def bytes2image(bytes):
    image = cv2.imdecode(np.frombuffer(bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def savefile(path, file: FileStorage):
    file.save(path)


def savefiles(root, files: List[FileStorage]):
    file_utils.create_dir(root)
    outpath = []
    for file in files:
        if not file.filename: continue
        path = os.path.join(root, file.filename)
        print(f"save file:{path}")
        savefile(path=path, file=file)
        outpath.append(path)
    return outpath
