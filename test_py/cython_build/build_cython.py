# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-09-20 14:35:22
    @Brief  :
"""

import os
from pybaseutils import file_utils
from pybaseutils.build_utils import cython_utils

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
    exclude_dirs = ['.git', '.idea', 'docs', 'test', 'build', 'dist']
    cython_utils.build_cython(root, app, build, exclude_dirs=exclude_dirs, exclude_files=exclude_files)
