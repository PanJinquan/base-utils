#!/usr/bin/env bash
#sudo docker build -t="docker.dm-ai.cn/algorithm-research/calligraphy/py36-trt8.4-torch17:v1.2" .
sudo docker run -it --gpus all -p 8000:80 -v `pwd`:/app docker.dm-ai.cn/algorithm-research/calligraphy/py36-trt8.4-torch17:v1.2 /bin/bash

