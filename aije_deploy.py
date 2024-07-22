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
import subprocess
from pybaseutils import file_utils
from pybaseutils.build_utils import pyarmor_utils


def run_command(comd: list):
    for c in comd:
        print(">>>>", c)
        process = subprocess.run(c, shell=True)
        if process.returncode != 0:
            print("Command executed with errors or did not execute at all.")
            print("Error output:", process.stderr)
            exit()


if __name__ == '__main__':
    aije_root = "/home/PKing/nasdata/release/AIJE/aije-release"
    url = "https://gitlab.dm-ai.cn/aije/aije-algorithm"
    runtime = "/home/PKing/nasdata/release/AIJE/aije-algorithm/aije-algorithm-deployment/.runtime-structure.json"
    services = file_utils.read_json_data(runtime)["services"]
    for service in services:
        tag = list(service["versions"].keys())[0]
        name = service["name"]
        path = os.path.join(aije_root, name)
        repo = os.path.join(url, name)
        tool = os.path.join(aije_root, "build_pyarmor.py")
        comd = []
        if not os.path.exists(path):
            comd.append(f"cd {aije_root} && git clone {repo}")
        comd.append(f"cd {path} && git checkout obfuscation && git reset --hard {tag}")
        comd.append(f"cp {tool} ./")
        comd.append(f"pwd && python build_pyarmor.py")
        run_command(comd)
        exit(0)
