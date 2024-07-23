#!/usr/bin/env bash
#--------代码混淆方法使用方法-----------------
# pip install pyarmor==7.7.4
# https://www.jianshu.com/p/c1d3d79e3545/
# 开发人员在各自分支开发任务(dev-pjq),开发完成后，统一在dev分支合并代码，确保在dev环境正常
# bash deploy.sh # 运行代码混淆脚本处理
git checkout dev && git pull # 拉取dev最新代码
git checkout obfuscation # 切换到代码混淆分支
git reset --hard dev # 将dev分支的最新代码强行覆盖obfuscation分支
python build_pyarmor.py # 执行代码混淆脚本
cp -r  build/* ./
rm -rf build
git add . && git commit -m "增加混淆"
git push origin obfuscation --force # 提交代码混淆
echo "请在gitlab仓库的obfuscation分支新建tag部署stage环境"
echo "conda activate pytorch-py38 &&PORT=40001 python app/main.py"
