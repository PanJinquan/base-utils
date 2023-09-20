# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-09-20 14:35:22
    @Brief  : pip install pyarmor==7.7.4
              https://pyarmor.readthedocs.io/en/v5.4.0/project.html#managing-obfuscated-scripts-with-project
              https://baijiahao.baidu.com/s?id=1775756280786745996&wfr=spider&for=pc
              https://www.jianshu.com/p/c1d3d79e3545/
"""

import os
from pybaseutils import file_utils
from pybaseutils.build_utils import cython_utils

IGNORE_DIRS = cython_utils.IGNORE_DIRS


def build_pyarmor_project(root,
                          entry,
                          build="./build",
                          exclude_dirs=IGNORE_DIRS
                          ):
    """
    rm -rf .pyarmor_config
    rm -rf dist
    pyarmor init --entry=app/main.py
    pyarmor build
    :param root: 项目根目录路径
    :param entry: 主函数
    :param build: 编辑输出目录
    :param exclude_dirs: 不需要处理的文件夹
    :param exclude_files: 不需要处理的文件
    :return:
    """
    file_utils.copy_dir(root, build, exclude=exclude_dirs)
    os.system(f'pyarmor --version')
    os.system(f'rm -rf .pyarmor_config')
    os.system(f'rm -rf dist')
    os.system(f'pyarmor init --entry={entry}')
    os.system(f'pyarmor build')
    os.system(f'cd dist && cp -r pytransform ./app')
    file_utils.copy_dir(os.path.join(root, "dist"), build, exclude=[])


def build_pyarmor_bk(root,
                     app="./app",
                     main="app/main.py",
                     build="./build",
                     exclude_dirs=IGNORE_DIRS
                     ):
    """
    存在BUG，不推荐使用
    :param root: 项目根目录路径
    :param main: 主函数
    :param build: 编辑输出目录
    :param exclude_dirs: 不需要处理的文件夹
    :param exclude_files: 不需要处理的文件
    :return:
    """
    # file_utils.copy_dir(root, build, exclude=exclude_dirs)
    os.system(f'pyarmor --version')
    os.system(f'pyarmor obfuscate --recursive {main}')
    os.system(f'cd dist && cp -r pytransform ./app')
    # file_utils.copy_dir(os.path.join(root, "dist"), build, exclude=[])


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    build = os.path.join(root, "build")
    entry = "app/main.py"
    build_pyarmor_project(root, entry, build)
