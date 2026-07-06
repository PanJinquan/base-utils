# Docker使用方法

- 国内镜像：
- docker-0.unsee.tech
- docker.1ms.run
- dhub.kubesre.xyz

## 安装

- 安装 https://blog.csdn.net/guyuealian/article/details/100078444

```bash
sudo apt-get update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

- 这支持GPU版本

```bash
# https://bbs.deepin.org/zh/post/262226
# https://bbs.deepin.org/zh/post/193717
# 有nvidia显卡，安装了cuda，需要补充安装
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y kmod
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 将用户添加到docker用户组

```bash
# 创建docker用户组
sudo groupadd docker
# 把当前用户加入docker用户组
sudo usermod -aG docker $USER
# 更新激活docker用户组
newgrp docker
# 验证不需要sudo执行docker命令
docker run hello-world

```

## 加速 Docker 镜像的下载速度

- docker pull hello-world --registry-mirror=https://docker.m.daocloud.io (临时方法)
- 打开或创建 Docker 配置文件 /etc/docker/daemon.json (永久配置)
- 添加以下内容：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.imgdb.de",
    "https://docker-0.unsee.tech",
    "https://docker.hlmirror.com",
    "https://docker.1ms.run"
  ]
}
```

- 保存文件并重启 Docker 服务： sudo systemctl daemon-reload && sudo systemctl restart docker
- 验证配置是否生效: docker pull hello-world


## 阿里云docker镜像仓库
- 地址： https://cr.console.aliyun.com/cn-guangzhou/instance/namespaces
```bash
# 1. 登录阿里云 Container Registry
aliyun=crpi-r7ny3w7dyvydm6vb.cn-guangzhou.personal.cr.aliyuncs.com
docker login --username=390737991@qq.com $aliyun
#2. 从Registry中拉取镜像
docker pull $aliyun/python-image-rep/py3.10-cuda11.7-cudnn8.5-torch2.0:[镜像版本号]
# 3. 将镜像推送到Registry
docker tag [ImageId] $aliyun/python-image-rep/py3.10-cuda11.7-cudnn8.5-torch2.0:[镜像版本号]
docker push $aliyun/python-image-rep/py3.10-cuda11.7-cudnn8.5-torch2.0:[镜像版本号]

```


## 镜像操作


```bash
# 登录docker: 
docker login hub.docker.com
docker login docker.dm-ai.cn
docker login --username=390737991@qq.com crpi-r7ny3w7dyvydm6vb.cn-guangzhou.personal.cr.aliyuncs.com # 登录阿里云docker镜像仓库68
# 查看所有镜像
docker images
# 查看所有容器
docker ps  -a 

# 启动、停止、重启、进入容器命令
docker start container_name/container_id
docker stop container_name/container_id
docker restart container_name/container_id
docker attach container_name/container_id # 后台启动一个容器后，如果想进入到这个容器，可以使用attach命令
# 删除所有停止的容器、所有未被使用的网络、所有悬空镜像、所有构建缓存
docker system prune -a
```

## 制作镜像

```bash
docker commit -m "info" -a "panjinquan" container_id $image
docker push image_ids:tag
docker tag old-image new-image # 修改镜像名称
```

## 镜像Python安装方法

```bash
apt update
# 使用add-apt-repository命令来添加包含Python 3.10的deadsnakes PPA
apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa  && apt update
apt install -y wget python3.10 python3.10-dev python3.10-venv python3.10-distutils
ln -s /usr/bin/python3.10 /usr/bin/python 
# apt install python3-pip  # 可能安装的不是python3.10
wget http://mirrors.aliyun.com/pypi/get-pip.py && python3.10 get-pip.py
#apt install python3.10-distutils && wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py
# 设置默认python,通过whereis python3.10
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 镜像安装opencv依赖库

```bash
apt update
# 安装一些基础的编译工具和依赖库
apt install -y build-essential cmake git pkg-config
# 支持图像处理和视频输入输出
apt install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt install -y libxvidcore-dev libx264-dev libdc1394-22-dev libgphoto2-dev libgdal-dev
# OpenCV支持图形用户界面（GUI），需要安装GTK库   
apt install -y libgtk-3-dev  
# 安装一些用于并行处理和数学运算优化的库
apt install -y libtbb-dev libatlas-base-dev gfortran 
```

#### nvidia基础镜像

- nvidia-docker镜像：https://hub.docker.com/r/nvidia/cuda
- 国内镜像： https://dockers.xuanyuan.me/image/nvidia/cuda ，如果要拉镜像，则使用docker-0.unsee.tech/{原始镜像名称}
- https://dockers.xuanyuan.me/ docker.1ms.run/{原始镜像名称}

```bash
#TODO 可以添加--name $name 指定容器别名
#image="docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:latest"
#image="docker.dm-ai.cn/algorithm-research/llama2-chinese:latest"
image="docker.dm-ai.cn/algorithm-research/panjinquan/py3.10-cuda11.7-cudnn8.5-torch2.0:llm"
image="docker.dm-ai.cn/algorithm-research/panjinquan/py3.10-cuda11.7-cudnn8.5-torch2.0:llm-v2"
image="docker.dm-ai.cn/algorithm-research/panjinquan/py3.10-cuda11.7-cudnn8.5-torch2.0:llm-v2.1"
image="docker.dm-ai.cn/algorithm-research/panjinquan/py3.10-cuda11.7-cudnn8.5-torch2.0:llm-gradio"
image="docker.dm-ai.cn/algorithm-research/panjinquan/py3.10-cuda11.7-cudnn8.5-torch2.0:dev" # 推荐这个
image="crpi-r7ny3w7dyvydm6vb.cn-guangzhou.personal.cr.aliyuncs.com/python-image-rep/py3.10-cuda11.7-cudnn8.5-torch2.0:llm"

#docker build -t=$image .
#docker run -it --gpus all -p 7860:7860 -v `pwd`:/app $image /bin/bash
docker run -it --gpus all -p 7860:7860 --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap=-1 --memory=256G --runtime=nvidia --ipc host --privileged --network host  -v `pwd`:/app $image /bin/bash

# TODO  tensorrt基础镜像
image="glenaaa/tensorrt-ubuntu20.04-cuda12.1:v0.6" # python 3.11.5,cuda 12.1  测试可用
image="docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-py3.11-cuda12.1-cudnn8.9-torch2.5-trt8.6:base"


# cuda12.1 https://hub.docker.com/r/nvidia/cuda/tags?name=12.1
docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 # nvidia-docker基础镜像,无python


# TODO
docker pull nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04 # nvidia-docker基础镜像,无python
image=nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04 # nvidia-docker基础镜像,无python
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:base
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1-trt8.2
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1-trt8.4
# TODO
docker pull docker-0.unsee.tech/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 # nvidia-docker基础镜像,无python
image=docker.dm-ai.cn/algorithm-research/panjinquan/cuda-11.3.1-cudnn8-devel-ubuntu20.04:base # nvidia-docker基础镜像,无python
image=docker.dm-ai.cn/algorithm-research/panjinquan/cuda-11.3.1-cudnn8-devel-ubuntu20.04-py3.10:base # nvidia-docker基础镜像,python3.10
image=docker.dm-ai.cn/algorithm-research/panjinquan/cuda-11.3.1-cudnn8-devel-ubuntu20.04-py3.10:ncnn # nvidia-docker基础镜像,python3.10,ncnn模型推理
image=docker.dm-ai.cn/algorithm-research/panjinquan/cuda-11.3.1-cudnn8-devel-ubuntu20.04-py3.10:torch2.1.0

# docker pull pavelogurthov/vulkan vulkan在CUDA12.X测试正常，但在CUDA11.X异常
docker pull docker-0.unsee.tech/randomgraphics/vulkan:11.2.182-cuda-11.3.1-ubuntu-20.04 
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-cuda11.3.1-vulkan11.2.182:base
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-cuda11.3.1-vulkan11.2.182-py3.10:base
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-cuda11.3.1-vulkan11.2.182-py3.10:ncnn # CUDA12.X  vulkan 1.2.182

# docker pull docker-0.unsee.tech/nvidia/vulkan:1.3-470 # vulkan测试异常
# docker pull docker-0.unsee.tech/nvidia/vulkan:1.2.170-470 # 但在CUDA11.X正常,CUDA12.X测试异常，
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-nvidia470-vulkan1.2.170:base
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-nvidia470-vulkan1.2.170-py3.10:base
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-nvidia470-vulkan1.2.170-py3.10:ncnn # CUDA11.X

# vulkan 测试正常
docker pull docker.dm-ai.cn/public/ubuntu2004-cuda114:1.0
 
# TODO 运行镜像
docker run -it --gpus all -p 40000:80 $image /bin/bash
docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash
```

## docker清除垃圾

```bash
du -ah --max-depth=1/                            # 查看文件大小
 
apt autoclean && apt clean && apt autoremove -y
pip cache purge # 清除 pip 缓存中的所有文件。
rm -rf ~/.cache
conda clean --all #删除conda无用的包和缓存
```

## 一些异常处理

- Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

> 解决方法：https://zhuanlan.zhihu.com/p/373493938?utm_id=0

- libgl.so.1: cannot open shared object file

> 解决方法：https://blog.csdn.net/weixin_42990464/article/details/125203404

```bash
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# 修改完成后，让配置生效
ldconfig
```

- Error： This might be caused by insufficient shared memory (shm).

> 解决方法：https://www.hangge.com/blog/cache/detail_3221.html <br/>

```bash
#（1）一种方法就是在容器启动命令上添加 --shm-size 参数，增加 shm 大小
docker run --shm-size = 256m ...
#（2）另一种方法就是启动命令上添加 --ipc=host，让容器与主机共享内存。
docker run --ipc=host ...
```

## docker添加环境变量失效问题

- 建议在 ~/.bashrc设置，不要在/etc/profile配置

## 删除none的镜像

```bash

# 1. 先删除所有已停止的容器
docker container prune -f
# 2. 然后再删除无标签的镜像
docker images | grep none | awk '{print $3}' | xargs docker rmi

docker image prune -f # 删除所有悬空镜像（即没有被容器引用且 tag 为 <none> 的镜像）
docker ps -a | grep "Exited" | awk '{print $1}' | xargs docker stop
docker ps -a | grep "Exited" | awk '{print $1}' | xargs docker rm
docker images | grep none | awk '{print $3}' | xargs docker rmi
```