#!/bin/bash

# 检查是否在Git仓库中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "错误：当前目录不是Git仓库"
    exit 1
fi

# 检查是否已安装Git LFS
if ! git lfs --version > /dev/null 2>&1; then
    echo "错误：Git LFS未安装，请先安装Git LFS"
    echo "安装指南：https://git-lfs.github.com/"
    exit 1
fi

# 初始化Git LFS（如果尚未初始化）
git lfs install

# 查找大于100MB的文件
echo "正在查找大于100MB的文件..."
large_files=$(find . -type f -size +100M -not -path "./.git/*" | sed 's|^\./||')
echo "找到以下大于100MB的文件："
echo "$large_files"
echo ""

# 将大文件添加到Git LFS跟踪
echo "正在将这些文件添加到Git LFS跟踪..."
for file in $large_files; do
    git lfs track "$file"
done

# 更新.gitattributes,提交更改
git add .gitattributes
git commit -m '将大文件迁移到Git LFS'

echo ""
echo "已将大文件添加到Git LFS跟踪，请执行以下操作完成迁移："
echo ""
echo "推送更改到远程仓库："
echo "   git push origin <分支名>"
echo ""
echo "注意：如果这些大文件已经在之前的提交中存在，你可能需要使用"
echo "git lfs migrate 命令来重写历史记录。"