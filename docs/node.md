# Node.js
没有 sudo 权限时，可以通过以下几种方式安装 Node.js、npm 和 pfnpm：

## 方法一：使用 Node Version Manager (NVM) - 推荐 👍

NVM 是专门为这种情况设计的，不需要 sudo 权限。

### 1. 安装 NVM
```bash
# 下载并安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash

# 或者使用 wget
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash
```

### 2. 重新加载 shell 配置
```bash
# 重新加载 bash 配置
source ~/.bashrc

# 或者如果是 zsh
source ~/.zshrc

# 或者直接重新登录
exit
# 重新登录服务器
```

### 3. 验证 nvm 安装
```bash
nvm --version
```

### 4. 安装 Node.js（包含 npm）
```bash
# 安装最新的 LTS 版本
nvm install --lts

# 或者安装特定版本
nvm install 22.0.0

# 使用安装的版本
nvm use 22.0.0

# 设置为默认版本
nvm alias default 22.0.0
```

### 5. 验证安装
```bash
node --version
npm --version
```

### 6. 安装 pnpm
```bash
# 使用 npm 安装 pnpm
npm install -g pnpm

# 或者使用 nvm 的独立方式（推荐）
curl -fsSL https://get.pnpm.io/install.sh | sh

# 验证 pnpm
pnpm --version
```

## 方法二：手动下载二进制文件

### 1. 下载 Node.js 二进制包
```bash
# 创建安装目录
mkdir -p ~/local
cd ~/local

# 下载 Node.js（替换为最新版本号）
wget https://nodejs.org/dist/v18.16.0/node-v18.16.0-linux-x64.tar.xz

# 解压
tar -xf node-v18.16.0-linux-x64.tar.xz

# 重命名方便使用
mv node-v18.16.0-linux-x64 nodejs
```

### 2. 配置环境变量
编辑 `~/.bashrc` 或 `~/.bash_profile`：
```bash
nano ~/.bashrc
```

添加以下内容：
```bash
export PATH="$HOME/local/nodejs/bin:$PATH"
```

### 3. 使配置生效
```bash
source ~/.bashrc
```

### 4. 验证安装
```bash
node --version
npm --version
```

### 5. 安装 pnpm
```bash
npm install -g pnpm
```

## 方法三：使用 conda/miniconda（如果服务器有安装）

```bash
# 安装 Node.js
conda install -c conda-forge nodejs

# 安装 pnpm
conda install -c conda-forge pnpm
```

## 配置技巧

### 设置 npm 全局安装目录（避免权限问题）
```bash
# 在用户目录创建 npm 全局目录
mkdir ~/.npm-global

# 配置 npm 使用此目录
npm config set prefix '~/.npm-global'

# 将路径添加到 PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 验证完整安装
```bash
# 检查所有工具版本
echo "Node.js版本: $(node --version)"
echo "npm版本: $(npm --version)" 
echo "pnpm版本: $(pnpm --version)"

# 测试安装包
npm init -y
pnpm init
```

## 推荐方案

**首选 NVM 方案**，因为：
- 专门为无 sudo 权限设计
- 可以轻松切换多个 Node.js 版本
- 自动处理环境变量配置
- 社区支持好，问题容易解决

如果遇到网络问题，可以尝试方法二手动下载二进制包。安装完成后，你就可以正常使用这些工具进行开发了！