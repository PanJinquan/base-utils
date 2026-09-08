# vulkan

- 下载驱动：https://vulkan.lunarg.com/sdk/home
- 查询版本：https://vulkan.gpuinfo.org/listdevices.php 查询显卡支持的vulkan版本
- 参考这个安装vulkan: 在Ubuntu 20.04上从零编译MNN（含Vulkan加速配置） https://blog.csdn.net/weixin_29053383/article/details/159311826 

## 安装

```bash
# 下载安装包
https://sdk.lunarg.com/sdk/download/1.3.236.0/linux/vulkansdk-linux-x86_64-1.3.236.0.tar.gz
https://sdk.lunarg.com/sdk/download/1.2.182.0/linux/vulkansdk-linux-x86_64-1.2.182.0.tar.gz
https://sdk.lunarg.com/sdk/download/1.3.280.0/linux/vulkansdk-linux-x86_64-1.3.280.0.tar.xz
# 安装工具
sudo apt install vulkan-tools

cp /app/docs/vulkansdk-linux-x86_64-1.3.236.0.tar.gz /tmp/ && cd /tmp/
tar -zxvf vulkansdk-linux-x86_64-1.3.236.0.tar.gz
# TODO 配置环境变量 gedit ~/.bashrc 然后 source ~/.bashrc
VULKAN_SDK=/home/PKing/app/vulkansdk-1.3.280.0/x86_64
PATH=${VULKAN_SDK}/bin:${PATH}
LD_LIBRARY_PATH=${VULKAN_SDK}/lib:${LD_LIBRARY_PATH}
VK_LAYER_PATH=${VULKAN_SDK}/etc/vulkan/explicit_layer.d

# TODO 验证
vulkaninfo --summary
vkcube

# TODO 如果运行出现：libvulkan.so找不到，请在sudo nano /etc/ld.so.conf.d/vulkan-custom.conf添加
/home/PKing/app/vulkansdk-1.3.280.0/x86_64/lib
```

docker pull randomgraphics/vulkan:1.2.162


## 安装2 (未测试)

```bash
# 如果你需要运行 32 位 Vulkan 应用（比如某些游戏），请先启用 i386 架构
sudo dpkg --add-architecture i386
# 安装Intel / AMD 开源显卡驱动用户
sudo apt install mesa-vulkan-drivers mesa-vulkan-drivers:i386
# 安装Vulkan工具
sudo apt install vulkan-tools
sudo apt install libvulkan1
sudo apt install libvulkan-dev
sudo apt install mesa-vulkan-drivers
# 或者搜索vulkan相关的工具包
# 验证
vulkaninfo --summary
vkcube
```

## 安装 (RK3588)

```bash
sudo apt update
sudo apt install vulkan-tools libvulkan1 libvulkan-dev
# 或者搜索vulkan相关的工具包
# 验证
vulkaninfo --summary
vkcube
# 对于AMD或Intel集成显卡，通常使用开源的Mesa驱动。安装以下包即可
sudo apt install mesa-vulkan-drivers
```

