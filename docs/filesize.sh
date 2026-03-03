#!/usr/bin/env bash
# TODO 统计当前目录下每个文件夹的空间大小
#du -h --max-depth=1
#du -h -d 1
#du -h --max-depth=1 | sort -hr # 按大小排序显示
du -h --max-depth=1 | sort -hr > filesize.txt # 按大小排序显示