# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-09-20 14:35:22
    @Brief  : pip install pyarmor==7.7.4
              https://pyarmor.readthedocs.io/en/v5.4.0/project.html#managing-obfuscated-scripts-with-project
              https://pyarmor.readthedocs.io/en/v7.3.0/project.html#project-configuration-file
              https://pyarmor.readthedocs.io/zh/v8.5.0/part-1.html
              https://docs.python.org/2/distutils/sourcedist.html#commands (Commands)
              https://baijiahao.baidu.com/s?id=1775756280786745996&wfr=spider&for=pc
              https://www.jianshu.com/p/c1d3d79e3545/
"""

import os
from pybaseutils import file_utils

IGNORE_DIRS = ['.git', '.idea', 'docs', 'build', 'dist']


def build_pyarmor_project(root,
                          entry,
                          build="./build",
                          manifest="global-include *.py",
                          exclude_dirs=IGNORE_DIRS
                          ):
    """
    rm -rf .pyarmor_config
    rm -rf dist
    pyarmor init --entry=app/main.py
    pyarmor build
    :param root: 项目根目录路径
    :param entry: 主函数/入口函数
    :param build: 编译输出目录
    :param manifest: 用于指定需要处理的文件，默认global-include *.py
              prune  ：去除某个目录的所有文件
              exclude：去除某个文件
    :param exclude_dirs: 不需要处理的文件
    :return:
    """
    app = os.path.dirname(entry)
    file_utils.copy_dir(root, build, exclude=exclude_dirs)
    os.system(f'pyarmor --version')
    os.system(f'rm -rf .pyarmor_config')
    os.system(f'rm -rf dist')
    os.system(f'pyarmor init --entry={entry}')
    if manifest:
        os.system(f'pyarmor config --manifest="{manifest}"')
    os.system(f'pyarmor build --output={build} --force')
    # os.system(f'cd {build} && cp -r pytransform {os.path.join(build, app)}')
    file_utils.move_dir(os.path.join(build, "pytransform"), os.path.join(build, app, "pytransform"))
    file_utils.remove_dir(os.path.join(build, "pytransform"))
    # 将pyarmor编译结果保存到
    # file_utils.copy_dir(os.path.join(root, "dist"), build, exclude=[])


def build_pyarmor_bk(root,
                     app="./app",
                     main="app/main.py",
                     build="./build",
                     exclude_dirs=IGNORE_DIRS
                     ):
    """
    存在BUG，不推荐使用
    :param root: 项目根目录路径
    :param main: 主函数/入口函数
    :param build: 编译输出目录
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
