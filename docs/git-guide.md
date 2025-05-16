# GIT教程

- Git 全局设置
```bash
git config --global user.name "Jinquan"
git config --global user.email "Jinquan"
```


- 解决git pull/push需要输入密码的问题
  https://zhuanlan.zhihu.com/p/537646478

```bash
git config --global credential.helper store
```

- 将一个分支完全覆盖(不是合并)到另一个分支

  案例: 将dev分支的代码完全覆盖到master上
```bash
git checkout master
git reset --hard dev
git push origin master --force
```

- fatal: unable to access 'https://*****.git': Failed connect to 127.0.0.1:8888; Connection refused

  取消代理即可：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy  
```

- 两个不同仓库进行合并
  
```bash
# 本地仓库：Pytorch-YOLOv8
# 添加另一个仓库作为远程仓库：git remote add other-repo url/to/other-repo
git remote add ultralytics https://githubfast.com/ultralytics/ultralytics

# 拉取另一个仓库的更改:git fetch other-repo
git fetch ultralytics

# 创建并切换到新分支：git checkout -b other-repo-branch other-repo/master
git checkout -b ultralytics ultralytics/main
git push origin ultralytics

# 切换回目标仓库的主分支
git checkout main

# 合并新分支到目标仓库的主分支main
git merge other-repo-branch --allow-unrelated-histories
 
# 如果一切顺利，你可以将合并后的更改推送到远程目标仓库
git push origin main

```

- 查找大文件
```bash
# 查找大于100MB的文件
large_files=$(find . -type f -size +100M -not -path "./.git/*" | sed 's|^\./||')
echo "$large_files"
```
- 方法1：LFS上传大文件(>100M)，参考[lfs.sh](lfs.sh)
- 方法2：git lfs migrate import --above=100MB --everything # 将历史提交中所有大于100MB的文件迁移到LFS管理
- 恢复大文件
```bash
git lfs install       # 初始化 LFS
git lfs fetch origin  # 下载所有 LFS 文件
git lfs checkout      # 将文件还原到工作区
```



- 从GitHub迁移到GitLab
  
```bash
# 先拉所有分支
git branch -r | grep -v 'HEAD' | while read branch; do 
  git checkout -b ${branch#origin/} $branch
done
# 确保本地是最新的
git fetch --all
git pull --all
# 迁移所有大于 100MB 的文件
# git lfs migrate import --above=100MB --everything
# git remote rename origin old-origin
git remote add origin https://gitcode.com/ai-sdk/Pytorch-Segment-Trainer.git
# 如果出现错误：远程origin已经存在，则需要添加多仓库推送
# git remote set-url --add origin https://gitcode.com/ai-sdk/Pytorch-Segment-Trainer.git
git push -u origin --all
git push -u origin --tags
```


- 推送现有的文件
    
```bash
cd existing_folder
git init
git remote add origin https://gitcode.com/PKing/cv-sdk-tnn.git
git add .
git commit -m "Initial commit"
git branch -m main
git push -u origin main
```
