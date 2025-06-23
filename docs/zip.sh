#!/usr/bin/env bash
# 获取当前目录下所有文件夹（不包括子目录里的文件夹）
for dir in */; do
    if [ -d "$dir" ]; then
        # 去掉末尾的斜杠，得到纯文件夹名
        dir_name="${dir%/}"
        # 压缩成 zip 文件
        echo "正在压缩: $dir_name → ${dir_name}.zip"
        zip -q -r "${dir_name}.zip" "$dir_name"
    fi
done

echo "所有文件夹压缩完成！"