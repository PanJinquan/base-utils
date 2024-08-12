#!/usr/bin/env bash
# 制作pip包： https://www.cnblogs.com/sting2me/p/6550897.html
# 发布pip包： https://packaging.python.org/tutorials/packaging-projects/
# PyPI recovery codes daa596d242587202 12fac17d7698478c 33553d161c290b60 e3f1a0684e48cefc a43eb1f0d0de6736 d377d440e6b45eb3 352ee03b086aa982 e648c218903f3b4c
# pip install twine
pip install dist/pybaseutils-*.*.*.tar.gz
twine upload dist/*  --verbose
#Enter your username: PKing
#Enter your password:

echo please use PIP to install pybaseutils:
echo pip install --upgrade pybaseutils -i https://pypi.org/simple