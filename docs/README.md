# 常用的工具使用方法

## memory profiler性能分析工具

- https://zhuanlan.zhihu.com/p/121003986
- 使用mprof run代替python demo.py，执行完成后，会生成一个 .dat 文件

```bash
mprof run demo.py 
```

- 要绘制内存在时间维度的使用情况，需要安装matplotlib，然后执行 mprof plot (直接执行会读取最新的 .dat 文件)：

```bash
mprof plot
mprof plot mprofile_20200329173152.dat # 指定绘制文件
mprof plot --flame mprofile_20200329173152.dat # 查看火焰图
```

## pip安装慢的问题

- 新建/修改: vim ~/.pip/pip.conf:

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
disable-pip-version-check = true
timeout = 120
```

## 远程挂载

- 挂载

```bash
sudo sshfs -o allow_other -o nonempty user@IP:/path/to/data  /path/to/local/data
```

- 解绑

```bash
fusermount -u /path/to/local/data # 解绑
```

- 软连接

```bash
ln -s source dist
```

## 中文解压乱码

- zip

```bash 
unzip -O CP936 xxx.zip # unzip -n test.zip -d /tmp
tar -zxvf xxx.tar.gz
```
- sudo apt-get install unar
> unar *.zip得到的文件

