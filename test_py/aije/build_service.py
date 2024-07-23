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
    runtime_data = file_utils.read_json_data(runtime)["services"]
    branch = "obfuscation"
    match = ["aije-equipment-detection"]
    use_tag = False  #
    # match = []
    for service in runtime_data:
        name: str = service["name"]
        if name.endswith("deployment"): continue
        if match and name not in match: continue
        tag = list(service["versions"].keys())[0]
        path = os.path.join(aije_root, name)
        repo = os.path.join(url, name)
        tool = os.path.join(aije_root, "build_pyarmor.py")
        comd = [f"echo obfuscation repository:{repo}"]
        if not os.path.exists(path):
            comd.append(f"cd {aije_root} && git clone {repo}")
        if use_tag:
            comd.append(f"cd {path} && git checkout dev && git pull && git checkout {branch} && git reset --hard {tag}")
            info = f"代码混淆:{tag}"
        else:
            comd.append(f"cd {path} && git checkout dev && git pull && git checkout {branch} && git reset --hard dev")
            info = f"代码混淆:dev"
        comd.append(f"cd {path} && cp {tool} ./")
        comd.append(f"cd {path} && python build_pyarmor.py && cp -r  build/* ./ && rm -rf build")
        comd.append(f"cd {path} && git add . && git commit -m '{info}' && git push origin {branch} --force ")
        # comd.append(f"cd {path} && PORT=40001 python app/main.py")
        run_command(comd)
        # exit(0)
