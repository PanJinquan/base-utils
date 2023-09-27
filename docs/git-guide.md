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
