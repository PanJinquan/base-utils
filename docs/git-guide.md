# GIT教程

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