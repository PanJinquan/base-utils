#!/usr/bin/env bash
#--------代码混淆方法使用方法-----------------
# pip install pyarmor==9.2.3
# https://www.jianshu.com/p/c1d3d79e3545/
# 开发人员在各自分支开发任务(dev-pjq),开发完成后，统一在dev分支合并代码，确保在dev环境正常
git checkout dev && git pull # 拉取dev最新代码
git checkout obfuscation # 切换到代码混淆分支
git reset --hard dev # 将dev分支的最新代码强行覆盖obfuscation分支
echo ----------------------------------------
ignore=(data docs build  dist) # 需要忽略拷贝的项目
exclude=(test
         app/config/config.py
         data docs build  dist
         )
echo ----------------------------------------
# exclude某个文件必须+./，否则无法排除
exclude=("${exclude[@]/#/./}") # 为exclude数组中的每个元素添加./前缀
for item in "${exclude[@]}"; do
  echo exclude: $item
done
echo ----------------------------------------
for item in "${ignore[@]}"; do
  echo ignore : $item
done
echo ----------------------------------------
outs=build # 混淆后的代码输出目录
dirs=($(ls -1)) # 将ls的输出保存到数组中
exclude="${exclude[*]}" # 空格分隔,转为字符串
#dirs=($(ls -a -1)) # 添加-a参数来显示所有文件，包括隐藏的
echo "遍历当前目录："
rm -rf $outs
rm -rf pyarmor_runtime_*
mkdir -p $outs
# TODO 拷贝项目代码到dirs目录
for dir in "${dirs[@]}"; do
  if [[ ! " ${ignore[@]} " =~ " ${dir} " ]]; then
    echo "$dir"
    cp -r $dir $outs
  fi
done

# TODO: 混淆项目代码
pyarmor gen  . --exclude "$exclude" --output $outs

echo ----------------------------------------
# TODO: 复制混淆后的代码到当前目录
cp -r  $outs/* ./
rm -rf $outs
#git add . && git commit -m "增加混淆"
#git push origin obfuscation --force # 提交代码混淆
#echo "请在gitlab仓库的obfuscation分支新建tag部署stage环境"
echo "PORT=40000 python app.py"
PORT=40000 python app.py