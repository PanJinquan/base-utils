# vulkan
- 下载驱动：https://vulkan.lunarg.com/sdk/home
- 查询版本：https://vulkan.gpuinfo.org/listdevices.php 查询显卡支持的vulkan版本

## 安装
```bash

# Ubuntu 20.04 (Focal Fossa) nvidia-3090
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-focal.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-focal.list
apt update
apt install vulkan-sdk

wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.309-jammy.list https://packages.lunarg.com/vulkan/1.4.309/lunarg-vulkan-1.4.309-jammy.list
apt update
apt install vulkan-sdk

wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.204-focal.list https://packages.lunarg.com/vulkan/1.3.204/lunarg-vulkan-1.3.204-focal.list
apt update && apt install vulkan-sdk

# 1.3.268.0
https://sdk.lunarg.com/sdk/download/1.3.236.0/linux/vulkansdk-linux-x86_64-1.3.236.0.tar.gz
https://sdk.lunarg.com/sdk/download/1.2.182.0/linux/vulkansdk-linux-x86_64-1.2.182.0.tar.gz

cp /app/docs/vulkansdk-linux-x86_64-1.3.236.0.tar.gz /tmp/ && cd /tmp/
tar -zxvf vulkansdk-linux-x86_64-1.3.236.0.tar.gz
VULKAN_SDK=/tmp/1.3.236.0/x86_64
PATH="${VULKAN_SDK}/bin:${PATH}"
LD_LIBRARY_PATH="${VULKAN_SDK}/lib:${LD_LIBRARY_PATH}"
VK_LAYER_PATH="${VULKAN_SDK}/etc/vulkan/explicit_layer.d"
vulkaninfo --summary
```

docker pull randomgraphics/vulkan:1.2.162