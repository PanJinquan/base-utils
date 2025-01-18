#!/usr/bin/env bash
# 制作pip包： https://www.cnblogs.com/sting2me/p/6550897.html
# 发布pip包： https://packaging.python.org/tutorials/packaging-projects/
# PyPI API token: pypi-AgEIcHlwaS5vcmcCJDk4MGFkZDMxLWViOTMtNDVjYi04MjMwLTc4ZjE1MmFhNGNlNwACKlszLCJiZTU2ZmZlOS0xMzk3LTQ0ZGEtOTQzZi0zYzE4OWJlZmRkOGQiXQAABiA0GWRqx1UHHE3dO6Mrhd21SBFO48QEZRldzFZJDX-l9A
# pip install twine
pip install dist/pybaseutils-*.*.*.tar.gz
twine upload dist/*  --verbose
#Enter your username: PKing
#Enter your password:

echo please use PIP to install pybaseutils:
echo pip install --upgrade pybaseutils -i https://pypi.org/simple
