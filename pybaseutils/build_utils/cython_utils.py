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


def build_app_modules(root,
                      app="app",
                      build="./_build",
                      exclude_files=['.git', '.idea', '_build', 'build'],
                      exclude_modules=[]):
    """
    获得项目需要编译的模块
    :param root: 项目根目录路径
    :param app: 项目app模型名称
    :param exclude_files: 不需要处理的文件
    :param exclude_modules: 不需要处理的模块
    :return: app_build, setup_file, modules
    """
    app_name = os.path.basename(root)
    app_build = os.path.join(build, app)
    file_utils.copy_dir(root, build, exclude=exclude_files)
    files = file_utils.get_files_lists(app_build, postfix=["*.py"], sub=False)
    files = file_utils.get_sub_list(files, dirname=os.path.dirname(app_build))
    modules = []
    for file in files:
        if file.startswith("./"): file = file[2:]
        if file.endswith('__init__.py') or (file in exclude_modules):
            continue
        src = os.path.join(build, file)
        dst = os.path.join(build, file + "x")
        shutil.copyfile(src, dst)
        modules.append(file[:-3])
    clear_app_build(app_build, delete_types=['.c', '.py', '.so'], exclude_file=exclude_modules)  # 只保留pyx文件
    setup_file = gen_app_setup_file(os.path.join(build, 'setup.py'), app_name, modules)
    return app_build, setup_file, modules


def clear_app_build(app_root, delete_types, exclude_file):
    """
    删除编译中间文件
    :param app_root:
    :param delete_types:
    :param exclude_file: 不需要处理的文件
    :return:
    """
    root = os.path.dirname(app_root)
    for parent, dirs, files in os.walk(app_root, topdown=False):
        if parent.endswith("__pycache__"):
            print("remove: {}".format(parent))
            shutil.rmtree(parent)
            continue
        for name in files:
            p = "." + name.split(".")[-1]
            path = os.path.join(parent, name)
            file = path[len(root) + 1:]
            if file.endswith('__init__.py') or (file in exclude_file): continue
            if p in delete_types:
                print("remove: {}".format(path))
                os.remove(path)


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    build_root = os.path.join(root, "_build")
    app = "app"
    # 利用cython把代码编译成so库(代码入口点和需要做反射的代码不能编译:如main,http接口)
    exclude_file = ['build_cpython.py',
                    'setup.py',
                    'app/main.py',
                    'app/service.py',
                    'app/utils/http_utils.py',
                    'app/routers/health.py',
                    'app/routers/interface.py',
                    'app/routers/version.py']  # 不进行编译的文件
    app_build, setup_file, modules = build_app_modules(root, app="app", build=build_root,
                                                       exclude_files=['.git', '.idea', '_build', 'build'],
                                                       exclude_modules=exclude_file)
    os.system(f'cd {build_root} && pwd && python3 {setup_file} build_ext --inplace')
    # clear_app_files
    # clear_app_build(app_build, delete_types=['.pyx', '.c'], exclude_file=exclude_file)  # 删除编译中间文件
    clear_app_build(app_build, delete_types=['.pyx', '.c', '.py'], exclude_file=exclude_file)  # 删除py文件和编译中间文件，用于部署
    # clear_app_build(app_build, delete_types=['.c', '.py'], exclude_file=exclude_file)  # 删除py文件和编译中间文件，用于部署
    # clear_app_build(app_build, delete_types=['.so', '.pyx', '.c'], exclude_file=exclude_file)  # 删除所有编译文件
