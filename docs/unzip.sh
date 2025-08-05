#!/usr/bin/env bash
#!/bin/bash

# 解压当前目录下所有 ZIP 文件的脚本
# 用法: ./unzip_all.sh [目标目录] (可选)

# 设置目标目录（如果未指定参数，则解压到各自 ZIP 文件名对应的目录）
DEST_DIR="${1:-./}"
# 遍历当前目录下的所有 ZIP 文件
for zipfile in *.zip; do
    if [ -f "$zipfile" ]; then
        echo "正在解压: $zipfile ..."

        # 如果未指定目标目录，则解压到以 ZIP 文件名（不含.zip）命名的目录
        if [ "$DEST_DIR" == "./" ]; then
            dir_name="${zipfile%.zip}"
            mkdir -p "$dir_name"
            unzip -O CP936 -qo "$zipfile" -d "$dir_name"
            echo "解压完成: $zipfile → $dir_name/"
        else
            # 解压到指定的目标目录
            mkdir -p "$DEST_DIR"
            unzip -O CP936 -qo "$zipfile" -d "$DEST_DIR"
            echo "解压完成: $zipfile → $DEST_DIR/"
        fi
    fi
done

echo "所有 ZIP 文件解压完成！"