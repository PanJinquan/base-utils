#!/usr/bin/env bash
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
#sudo apt-key add -
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#sudo tee /etc/apt/sources.list.d/nvidia-docker.list
#sudo apt-get update
#sudo apt-get install -y nvidia-container-toolkit
#sudo systemctl restart docker

#sudo docker build -t="docker.dm-ai.cn/algorithm-research/calligraphy/py36-trt8.4-torch17:v1.2" .
sudo docker run -it --gpus all -p 8000:80 -v `pwd`:/app docker.dm-ai.cn/algorithm-research/calligraphy/py36-trt8.4-torch17:v1.2 /bin/bash

