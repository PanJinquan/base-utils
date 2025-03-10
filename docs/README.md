# 常用的工具使用方法

## Linux常见命令

```bash
# xdg-open打开文件夹/文件
xdg-open ./
# 查找某个文件
find /usr -name libnvidia-ml* 

```

- tmux 快捷键

```bash
tmux ls # 查看有多少个窗口
Crtl + B + { 或 Crtl + Shift + Alt + B + { # 浏览LOG记录[
Crtl + B + S 或 Crtl + Shift + Alt + B + S # 浏览窗口
```

## memory profiler性能分析工具

- https://zhuanlan.zhihu.com/p/121003986
- 使用mprof run代替python demo.py，执行完成后，会生成一个 .dat 文件

```bash
mprof run demo.py 
```

- 要绘制内存在时间维度的使用情况，需要安装matplotlib，然后执行 mprof plot (直接执行会读取最新的 .dat 文件)：

```bash
mprof plot
mprof plot mprofile_20200329173152.dat # 指定绘制文件
mprof plot --flame mprofile_20200329173152.dat # 查看火焰图
```

## Ubuntu中监控CPU和GPU

- https://blog.csdn.net/qq_40078905/article/details/123087635
- pip install sysmon

## pip安装慢的问题

- 新建/修改: vim ~/.pip/pip.conf:

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
disable-pip-version-check = true
timeout = 120
```

- pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
- 若出现超时：pip install --default-timeout=1000000 -i https://pypi.tuna.tsinghua.edu.cn/simple
- 设置默认的镜像源：pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 远程挂载

- 挂载

```bash
sudo sshfs -o allow_other -o nonempty user@IP:/path/to/data  /path/to/local/data
```

- 解绑

```bash
sudo umount -l /Path/to/target # 解绑(推荐使用)
sudo fusermount -u  /Path/to/target # 解绑
```

- 软连接

```bash
ln -s source dist
```

## 文件解压和解压

```bash
zip -r fold.zip fold/    # 压缩fold文件夹
unzip -O CP936 fold.zip  # 解压fold.zip压缩文件(-O CP936可解决中文乱码问题)
tar -zxvf fold.tar.gz    # 解压fold.tar.gz文件
unar fold.zip            # 解压fold.zip压缩文件,解决中文乱码(安装：sudo apt-get install unar)
```

- zip分卷压缩文件

```bash
zip -r -s 3g fold.split.zip fold/
# -s 1g(或m)代表分卷大小GB,MB
# fold.split.zip为压缩包名
# fold/为待压缩的目录
```

- zip解压分卷文件

```bash
zip -s 0 fold.split.zip --out fold.zip
unzip fold.zip
```

- tar分卷压缩文件

```bash
tar cvzpf - fold | split -d -b 3078m - fold.tar.gz
# 其中 - myfile :输入文件夹名字; -b 2048m :每卷为2048m; - newfile :输出文件名
# 压缩完的文件命名为：fold.tar.gz00,fold.tar.gz01,fold.tar.gz03...
```

- tar解压分卷文件

```bash
cat fold*>fold.tar.gz   # 将分卷文件合并成一个压缩文件
tar xzvf fold.tar.gz    # 解压 
```

## 查看所有进程的命令ps aux

```bash
ps aux|grep python|grep -v grep
```

- grep python”的输出结果是，所有含有关键字“python”的进程，这是python程序
- grep -v grep”是在列出的进程中去除含有关键字“grep”的进程。

## 文件信息统计

```bash
# 查找某文件的位置使用whereis，例如：
whereis cuda # 查找cuda路径
whereis whereis cudnn_version # 查找cudnn路径
# 查看当前文件夹大小
du -ah --max-depth=1/
# 统计当前文件夹下文件的个数，包括子文件夹里的
ls -lR|grep "^-"|wc -l
# 统计文件夹下目录的个数，包括子文件夹里的
ls -lR|grep "^d"|wc -l
```

## 一些个性化别名

- 编辑：`vim ~/.bashrc`或`deepin-editor ~/.bashrc`
- 激活：`source ~/.bashrc`

```bash
# 服务路径
#alias ps -aux|grep redis
alias nasdata='echo /nasdata/atp/data/panjinquan'
alias cv='echo /atpcephdata/cv/panjinquan'
# pip 镜像源码(pip mirror=pipm)
alias pipm="echo -e 'https://pypi.tuna.tsinghua.edu.cn/simple\nhttps://pypi.org/simple'"
# 查看文件列表
alias ll='ls -l'
# 查看当前文件夹大小
alias dirsize='du -ah --max-depth=1/'
# 统计当前文件夹下文件的个数，包括子文件夹里的
alias countfile='ls -lR|grep "^-"|wc -l'
# 统计文件夹下目录的个数，包括子文件夹里的
alias countdir='ls -lR|grep "^d"|wc -l'
```

## 安装opencv-python常见的错误

- OpenCV+PyQT5兼容问题：https://www.sohu.com/a/602131072_121124366
- opencv-python4.2.0以上的版本，使用了qt库
- 解决方法1：

```bash
apt-get update
#apt-get upgrade
sudo apt-get install libopencv-dev python-opencv
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev    
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
```

- 解决方法2：使用pyqt5==5.13.2(亲测可用)

```bash
# 以下python3.8有效
# opencv-python==4.8.0.76 
# opencv-contrib-python==4.8.1.78 
# pyqt5==5.13.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.0 opencv-python==4.8.0.76 opencv-contrib-python==4.8.1.78 opencv-python-headless==4.8.0.76 pyqt5==5.13.2 # 新版labelme异常
# 可能需要安装的依赖包：
sudo apt-get update
sudo apt-get install -y \
    libxcb1-dev \
    libxcb-xinerama0-dev \
    libxcb-randr0-dev \
    libxcb-xtest0-dev \
    libxcb-shape0-dev \
    libxcb-xkb-dev \
    libxkbcommon-x11-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-keysyms1-dev \
    libxcb-render-util0-dev \
    qtbase5-dev \
    qttools5-dev

```

```bash
# 安装最新版
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python==4.11.0.86 opencv-python==4.11.0.86

PySide2==5.15.2.1

# opencv-python-headless
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PySide2
PyQt5==5.15.11 PyQt5-Qt5==5.15.16 PyQt5_sip==12.17.0

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
- 如果是github.com的地址，则直接替换为githubfast.com
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

## 表格

| Model   |     |      |
|---------|-----|------|
| YOLOv5s | 640 | 36.7 |
| YOLOv5m | 640 | 44.5 |


## 格式化

```python
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : ${YEAR}-${MONTH}-${DAY} ${HOUR}:${MINUTE}:${SECOND}
# @Brief  :
# --------------------------------------------------------
"""
```