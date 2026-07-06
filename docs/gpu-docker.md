# gpu-docker

- docker login hub.docker.com 
- docker login docker.dm-ai.cn


## nvidia基础镜像

- nvidia-docker镜像：https://hub.docker.com/r/nvidia/cuda
- 国内镜像： https://dockers.xuanyuan.me/image/nvidia/cuda ，如果要拉镜像，则使用docker-0.unsee.tech/{原始镜像名称}
- https://dockers.xuanyuan.me/ docker.1ms.run/{原始镜像名称}

## 常用的命令

```bash
# TODO 查询版本
du -ah --max-depth=1                             # 查看文件大小
cat /usr/local/cuda/version.txt                  # 查看cuda版本
cat /usr/include/cudnn_version.h                 # deb方式安装，查看cudnn版本
cat /usr/local/cuda/include/cudnn_version.h      # 库安装方式，查看cudnn版本
whereis cudnn_version.h                          # 查找方法
# TODO docker清除垃圾
apt autoclean && apt clean && apt autoremove -y
pip cache purge && pip3 cache purge              # 清除 pip 缓存中的所有文件。
rm -rf ~/.cache  && rm -rf /root/.cache          # 清除 ~/.cache 目录中的所有文件。
rm -rf /tmp/* /var/tmp/*                         # 删除临时文件和日志
rm -rf /var/lib/apt/lists/*                      # 删除apt缓存
conda clean --all                                # 删除conda无用的包和缓存
```

## 制作镜像

```bash
# TODO nvidia-docker镜像：https://hub.docker.com/r/nvidia/cuda
image=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 # nvidia基础镜像，无python
docker pull $image
docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash

# TODO 查看系统版本
cat /etc/os-release

# TODO 制作镜像
docker commit -m "info" -a "panjinquan" container_id $image
docker push image_ids:tag
docker tag old-image new-image # 修改镜像名称

# TODO 按需安装python3.10和python3-pip
apt update
apt install python3 -y
ln -s /usr/bin/python3.10 /usr/bin/python 
apt install python3-pip  # 可能安装的不是python3.10
# apt install -y wget
#wget http://mirrors.aliyun.com/pypi/get-pip.py && python3.10 get-pip.py
#apt install python3.10-distutils && wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py
# 设置默认python,通过whereis python3.10
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# TODO 安装opencv-python等基础开发库
apt-get update && apt-get install -y \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xv0 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6
pip install opencv-python opencv-contrib-python scikit-image scikit-learn
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:opencv

# TODO 安装pytorch等基础开发库
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8

# TODO 安装tensorrt等基础开发库
pip install tensorrt==8.6.1 pycuda # 目前适配版本
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8-tensorrt8.6

```

## docker镜像

```bash
image=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 # 基础镜像
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:opencv                 # 10G python opencv镜像
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8               # 17G torch2.8镜像，包含pytorch、torchvision等
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8-tensorrt8.6   # 28G torch2.8+tensorrt8.6镜像
image=docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8-onnx1.23      # 18G torch2.8+onnx1.23镜像


docker tag docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:opencv panjinquan/ubuntu22.04-cuda12.1-cudnn8-py310:opencv

```