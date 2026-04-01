# GIT教程

- Git 全局设置

```bash
git config --global user.name "PKing"
git config --global user.email "PKing"
# 自动转换CRLF为LF, 避免在Windows和Linux之间切换时出现问题
git config --global core.autocrlf input

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

- 从GitHub迁移到GitHub

```bash
# 先拉所有分支
git branch -r | grep -v 'HEAD' | while read branch; do 
  git checkout -b ${branch#origin/} $branch && git pull
done
# 确保本地是最新的
git fetch --all
git pull --all
git checkout dev-pjq
# 迁移所有大于 100MB 的文件
# git lfs migrate import --above=100MB --everything
git remote rename origin old-origin
git remote add origin https://gitcode.com/ai-sdk/Pytorch-Segment-Trainer.git
# 如果出现错误：远程origin已经存在，则需要添加多仓库推送
# git remote set-url --add origin https://gitcode.com/ai-sdk/Pytorch-Segment-Trainer.git
git push -u origin --all   # 如果文件太大，建议逐个分支推送： git push -u origin master
git push -u origin --tags  # 如果有标签，也需要推送
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

## 解决git pull/push需要输入密码的问题

- https://zhuanlan.zhihu.com/p/537646478

```bash
git config --global credential.helper store

```

## 解决Github克隆失败的问题

- 解决hugging face终端无法访问问题： https://zhuanlan.zhihu.com/p/676420788
- https://hf-mirror.com/
- https://blog.csdn.net/weixin_43431218/article/details/135403324
- https://blog.csdn.net/weixin_43431218/article/details/135544365
- 克隆github的仓库，请将github.com替换为githubfast.com
- 下载github的文件，则可以在这里下载：https://down.npee.cn/
- 如果是huggingface.co的地址，则直接替换为hf-mirror.com

```bash
# Linux
export HF_ENDPOINT=https://hf-mirror.com # 或者写入~/.bashrc中
# 如果要下载 https://huggingface.co/BAAI/DIVA/blob/main/OpenAICLIP/OpenAI-ViT-L-14-224.pth
# 则只需要把huggingface.co改为hf-mirror.com，即可在浏览器正常访问
./hfd.sh BAAI/DIVA --tool aria2c -x 4
```

## huggingface.co资源下载

- https://huggingface.co/
- 方法1：https://blog.csdn.net/gmmmmmmmm/article/details/135953651 (将下载连接huggingface.co`替换为 hf-mirror.com)
- 方法2：https://modelscope.cn/my/overview
- import huggingface_hub.constants 可以修改访问路径

```bash
# !pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

- 默认保持地址：~/.cache/huggingface/hub

```bash
from transformers import AutoImageProcessor, AutoModel
model_dir = "facebook/dinov2-base" # TODO 默认保持在~/.cache/huggingface/hub/models--facebook--dinov2-base
# TODO 如果移动到其他地方，则model_dir修改为path/to/models--facebook--dinov2-base-bk/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415
# TODO 即修改为preprocessor_config.json所在的根目录
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

```

## modelscope 资源下载

- 命令行中下载：

```bash
modelscope download --model Qwen/Qwen3-VL-2B-Instruct # 模型保持在 ~/.cache/modelscope/hub/models/
modelscope download --model OpenBMB/MiniCPM-V-2_6     # 模型保持在 ~/.cache/modelscope/hub/models/
modelscope download --model OpenBMB/MiniCPM-V-2_6 --local_dir /your/desired/path # 指定下载路径
# /nasdata/atp/data/panjinquan/Project/LLM/model/OpenBMB/MiniCPM-V-2_6
```