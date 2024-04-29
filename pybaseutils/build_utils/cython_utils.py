# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-09-20 14:35:22
    @Brief  :
"""

import os
import shutil
from pybaseutils import file_utils

IGNORE_DIRS = ['.git', '.idea', 'docs', 'test', 'build', 'dist']


def indent(num):
    assert isinstance(num, int)
    return ' ' * 4 * num


def gen_app_setup_file(setup_file, app_name, modules):
    """
    用于生成setup.py文件
    :param app_root:
    :param app_name:
    :param modules:
    :return:
    """
    setup_str = '''from Cython.Distutils import build_ext
import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
'''
    for m in modules:
        extension_str = indent(1) + 'Extension(\n'
        extension_str += indent(2) + '"{}",\n'.format(m).replace(os.sep, ".")
        extension_str += indent(2) + '["{}"],\n'.format(m + '.pyx')
        extension_str += indent(2) + 'include_dirs=[numpy_include]\n'
        extension_str += indent(1) + '),\n'
        setup_str += extension_str
    setup_str = setup_str[:-2]
    setup_str += '\n]\n\n'
    setup_str += "setup(name='{}',".format(app_name)
    setup_str += '''cmdclass={'build_ext': build_ext},ext_modules=cythonize(ext_modules, language_level="3"))
'''
    with open(setup_file, 'w') as f:
        f.write(setup_str)
    print("save setup.py in: {}".format(setup_file))
    return setup_file


def get_app_modules(root,
                    app,
                    build="./build",
                    exclude_dirs=IGNORE_DIRS,
                    exclude_files=[]):
    """
    获得项目需要编译的模块
    :param root: 项目根目录路径
    :param app: 项目app模型名称
    :param build: 编译输出目录
    :param exclude_dirs: 不需要处理的文件夹
    :param exclude_files: 不需要处理的文件
    :return: app_build, setup_file, modules
    """
    app_name = os.path.basename(root)
    app_build = os.path.abspath(os.path.join(build, app))
    file_utils.copy_dir(root, build, exclude=exclude_dirs)
    files = file_utils.get_files_lists(app_build, postfix=["*.py"], sub=False)
    files = file_utils.get_sub_list(files, dirname=os.path.dirname(app_build))
    modules = []
    for file in files:
        if file.startswith("./"): file = file[2:]
        if file.endswith('__init__.py') or (file in exclude_files):
            continue
        src = os.path.join(build, file)
        dst = os.path.join(build, file + "x")
        shutil.copyfile(src, dst)
        modules.append(file[:-3])
    clear_app_build(app_build, delete_types=['.c', '.py', '.so'], exclude_files=exclude_files)  # 只保留pyx文件
    setup_file = gen_app_setup_file(os.path.join(build, 'setup.py'), app_name, modules)
    return setup_file


def clear_app_build(app_build, delete_types, exclude_files):
    """
    删除编译中间文件
    :param app_build:
    :param delete_types:
    :param exclude_files: 不需要处理的文件
    :return:
    """
    root = os.path.dirname(app_build)
    for parent, dirs, files in os.walk(app_build, topdown=False):
        if parent.endswith("__pycache__"):
            print("remove: {}".format(parent))
            shutil.rmtree(parent)
            continue
        for name in files:
            p = "." + name.split(".")[-1]
            path = os.path.join(parent, name)
            file = path[len(root) + 1:]
            if file.endswith('__init__.py') or (file in exclude_files): continue
            if p in delete_types:
                print("remove: {}".format(path))
                os.remove(path)


def build_cython_setup(build, app, setup_file, exclude_files=[], clear=True):
    """
    :param build: 编辑输出目录
    :param app: 项目app模型名称
    :param setup_file:
    :param exclude_files: 不需要处理的文件
    :param clear: 是否清除编译中间文件
    :return:
    """
    app_build = os.path.abspath(os.path.join(build, app))
    tmp_build = os.path.join(build, "build")
    os.system(f'cd {build} && pwd && python3 {setup_file} build_ext --inplace')
    # clear_app_files
    if clear:
        if os.path.exists(tmp_build): shutil.rmtree(tmp_build)
        delete_types = ['.pyx', '.c', '.py']  # 删除py文件和编译中间文件,只留下.so文件，用于部署
        # delete_types = ['.so', '.pyx', '.c']  # 删除所有编译文件
        clear_app_build(app_build, delete_types=delete_types, exclude_files=exclude_files)


def build_cython(root,
                 app="./app",
                 build="./build",
                 exclude_dirs=IGNORE_DIRS,
                 exclude_files=[]):
    """
    :param root: 项目根目录路径
    :param app: 项目app模型名称
    :param build: 编译输出目录
    :param exclude_dirs: 不需要处理的文件夹
    :param exclude_files: 不需要处理的文件
    :return:
    """
    setup_file = get_app_modules(root, app=app, build=build,
                                 exclude_dirs=exclude_dirs,
                                 exclude_files=exclude_files)
    build_cython_setup(build, app, setup_file, exclude_files=exclude_files)


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    build = os.path.join(root, "build")
    app = "./app"
    # 利用cython把代码编译成so库(代码入口点和需要做反射的代码不能编译:如main,http接口)
    exclude_files = ['build_cython.py',
                     'setup.py',
                     'app/main.py',
                     'app/service.py',
                     'app/utils/http_utils.py',
                     'app/routers/health.py',
                     'app/routers/interface.py',
                     'app/routers/version.py']  # 不进行编译的文件
    exclude_dirs = IGNORE_DIRS
    build_cython(root, app, build, exclude_dirs=exclude_dirs, exclude_files=exclude_files)
