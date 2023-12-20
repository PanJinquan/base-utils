# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-09-20 14:35:22
    @Brief  : pip install pyarmor==7.7.4
              https://pyarmor.readthedocs.io/en/v5.4.0/project.html#managing-obfuscated-scripts-with-project
              https://pyarmor.readthedocs.io/en/v7.3.0/project.html#project-configuration-file
              https://docs.python.org/2/distutils/sourcedist.html#commands (Commands)
              https://baijiahao.baidu.com/s?id=1775756280786745996&wfr=spider&for=pc
              https://www.jianshu.com/p/c1d3d79e3545/
"""

import os
from pybaseutils import file_utils
from pybaseutils.build_utils import pyarmor_utils

IGNORE_DIRS = ['.git', '.idea', 'docs', 'build', 'dist']

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    build = os.path.join(root, "build")
    entry = "app/main.py"
    manifest = "global-include *.py," \
               "exclude test/*.py," \
               "exclude app/infercore/modules/indoor_component.py," \
               "exclude app/infercore/modules/outdoor_component.py," \
               "prune app/infercore/human_detector/yolov5/utils," \
               "prune app/infercore/human_pose/hrnet/yolov5/utils," \
               "prune app/infercore/equipment_detector/yolov5/utils," \
               "prune app/infercore/equipment_detector/yolov8/ultralytics,"
    pyarmor_utils.build_pyarmor_project(root, entry, build, manifest=manifest, exclude_dirs=IGNORE_DIRS)
