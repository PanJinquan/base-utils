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

## Ubuntu中监控CPU和GPU

- https://blog.csdn.net/qq_40078905/article/details/123087635
- pip install sysmon

## pip安装慢的问题

- 新建/修改: vim ~/.pip/pip.conf:

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
disable-pip-version-check = true
timeout = 120
```

- pip install --no-cache-dir opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

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

## 文件解压和解压

```bash
zip fold/ fold.zip       # 压缩fold文件夹
unzip -O CP936 fold.zip  # 解压fold.zip压缩文件(-O CP936可解决中文乱码问题)
tar -zxvf fold.tar.gz    # 解压fold.tar.gz文件
unar fold.zip            # 解压fold.zip压缩文件,解决中文乱码(安装：sudo apt-get install unar)
```

- zip分卷压缩文件

```bash
zip -r -s 3g fold.split.zip fold/
# -s 1g(或m)代表分卷大小GB,MB
# fold.split.zip为压缩包名
# fold/为待压缩的目录
```

- zip解压分卷文件

```bash
zip -s 0 fold.split.zip --out fold.zip
unzip fold.zip
```

- tar分卷压缩文件

```bash
tar cvzpf - fold | split -d -b 3078m - fold.tar.gz
# 其中 - myfile :输入文件夹名字; -b 2048m :每卷为2048m; - newfile :输出文件名
# 压缩完的文件命名为：fold.tar.gz00,fold.tar.gz01,fold.tar.gz03...
```

- tar解压分卷文件

```bash
cat fold*>fold.tar.gz   # 将分卷文件合并成一个压缩文件
tar xzvf fold.tar.gz    # 解压 
```

## 查看所有进程的命令ps aux

```bash
ps aux|grep python|grep -v grep
```

- grep python”的输出结果是，所有含有关键字“python”的进程，这是python程序
- grep -v grep”是在列出的进程中去除含有关键字“grep”的进程。

## 文件信息统计

```bash
# 查找某文件的位置使用whereis，例如：
whereis cuda # 查找cuda路径
whereis whereis cudnn_version # 查找cudnn路径
# 查看当前文件夹大小
du -ah --max-depth=1/
# 统计当前文件夹下文件的个数，包括子文件夹里的
ls -lR|grep "^-"|wc -l
# 统计文件夹下目录的个数，包括子文件夹里的
ls -lR|grep "^d"|wc -l
```