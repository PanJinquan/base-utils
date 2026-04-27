#!/usr/bin/env bash
#--------代码混淆方法使用方法-----------------
# pip install pyarmor==7.7.4
# https://www.jianshu.com/p/c1d3d79e3545/
# 开发人员在各自分支开发任务(dev-pjq),开发完成后，统一在dev分支合并代码，确保在dev环境正常
# bash deploy.sh # 运行代码混淆脚本处理
#git checkout dev && git pull # 拉取dev最新代码
#git checkout obfuscation # 切换到代码混淆分支
#git reset --hard dev # 将dev分支的最新代码强行覆盖obfuscation分支
# git reset --hard HEAD # 强制恢复到上一次commit
include=(
    'app/infercore'
    'app/utils'
)

exclude=(
#    'app/infercore1'
#    'app/infercore2'
)

for m in ${include[@]}; do
    echo "obfuscate ${m}"
    python obfuscator.py ${m} --exclude "${exclude[@]}"
done


# git add . && git commit -m "obfuscate code" && git push
