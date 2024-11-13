# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import os
import time
import re
import shutil
import numpy as np
import json
import glob
import random
import subprocess
import concurrent.futures
import numbers
import pickle
import argparse
import itertools
from datetime import datetime
from tqdm import tqdm

IMG_POSTFIX = ['*.jpg', '*.jpeg', '*.png', '*.tif', "*.JPG", "*.bmp"]
VIDEO_POSTFIX = ['*.mp4', '*.avi', '*.mov', "*.flv"]
AUDIO_POSTFIX = ['*.mp3', '*.wav']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_time(format="p"):
    """
    :param format:
    :return:
    """
    if format.lower() == "s":
        # time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    elif format.lower() == "p":
        # 20200508_143059_379116
        time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')
        time = time[:-2]
    else:
        time = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
    return time


def get_kwargs_name(**kwargs):
    prefix = []
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = [str(l) for l in v]
            prefix += v
        else:
            f = "{}_{}".format(k, v)
            prefix.append(f)
    prefix = "_".join(prefix)
    return prefix


def get_file_size(file, rate=1024):
    """获得文件大小"""
    return os.path.getsize(file) / rate


def replace_elements(items, src, dst, ignore=True):
    """
    将列表中，值为src改为dst
    :param items:
    :param src:
    :param dst:
    :param ignore: 是否提出空的数据
    :return:
    """
    for i, item in enumerate(items):
        if item == src: items[i] = dst
    if ignore: items = [it for it in items if str(it)]
    return items


def combine_flags(flags: list, use_time=True, info=True):
    """
    :param flags:
    :param info:
    :return:
    """
    out_flags = []
    for f in flags:
        if isinstance(f, dict):
            f = get_kwargs_name(**f)
        out_flags.append(f)
    if use_time:
        out_flags += [get_time()]
    out_flags = [str(f) for f in out_flags if f]
    out_flags = "_".join(out_flags)
    if info:
        print(out_flags)
    return out_flags


def write_file(file, data):
    """写二进制数据"""
    with open(file, 'wb') as f: f.write(data)


def read_file(file):
    """读取二进制数据"""
    with open(file, 'rb') as f: key = f.read()
    return key


class WriterTXT(object):
    """ write data in txt files"""

    def __init__(self, filename, mode='w'):
        self.f = None
        if filename:
            self.f = open(filename, mode=mode)

    def write_line(self, line, end='\n'):
        if self.f:
            self.f.write(line + end)
            self.f.flush()

    def write_line_list(self, line_list, end='\n'):
        if self.f:
            for line in line_list:
                self.write_line(line, end=end)
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def parser_classes(class_name):
    """
    class_dict = {class_name: i for i, class_name in enumerate(class_name)}
    :param class_name: filename,or list,dict
    :return: class_name,class_dict
    """
    if isinstance(class_name, str):
        class_name = read_data(class_name, split=None)
    elif isinstance(class_name, numbers.Number):
        class_name = [str(i) for i in range(int(class_name))]
    if isinstance(class_name, list):
        class_dict = {str(class_name): i for i, class_name in enumerate(class_name)}
    elif isinstance(class_name, dict):
        class_dict = class_name
    else:
        class_dict = None
    return class_name, class_dict


def read_json_data(json_path):
    """
    读取数据
    :param json_path:
    :return:
    """
    with open(json_path, 'rb') as f:
        json_data = json.load(f)
    return json_data


def write_json_path(json_file, json_data):
    """
    写入 JSON 数据
    :param json_file:
    :param json_data:
    :return:
    """
    with open(json_file, 'w', encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def write_data(filename, content_list, split=",", mode='w'):
    """保存list[list[]]的数据到txt文件
    :param filename:文件名
    :param content_list:需要保存的数据,type->list
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode=mode, encoding='utf-8') as f:
        for line_list in content_list:
            # 将list转为string
            line = "{}".format(split).join('%s' % id for id in line_list)
            f.write(line + "\n")
        f.flush()


def write_list_data(filename, list_data, mode='w'):
    """保存list[]的数据到txt文件，每个元素分行
    :param filename:文件名
    :param list_data:需要保存的数据,type->list
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode=mode, encoding='utf-8') as f:
        for line in list_data:
            # 将list转为string
            f.write(str(line) + "\n")
        f.flush()


def read_data(filename, split=",", convertNum=True):
    """
    读取txt数据函数
    :param filename:文件名
    :param split   :分割符
    :param convertNum :是否将list中的string转为int/float类型的数字
    :return: txt的数据列表
    Python中有三个去除头尾字符、空白符的函数，它们依次为:
    strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    lstrip：用来去除开头字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    注意：这些函数都只会删除头和尾的字符，中间的不会删除。
    """
    with open(filename, mode="r", encoding='utf-8') as f:
        content_list = f.readlines()
        content_list = [line.strip().strip('\ufeff').strip('\xef\xbb\xbf') for line in content_list]
        if split is None:
            content_list = [content.rstrip() for content in content_list]
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]
        if convertNum:
            for i, line in enumerate(content_list):
                line_data = []
                for l in line:
                    if is_int(l):  # isdigit() 方法检测字符串是否只由数字组成,只能判断整数
                        line_data.append(int(l))
                    elif is_float(l):  # 判断是否为小数
                        line_data.append(float(l))
                    else:
                        line_data.append(l)
                content_list[i] = line_data
    return content_list


def read_line_image_label(line_image_label):
    """
    line_image_label:[image_ids,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
    :param line_image_label:
    :return:
    """
    line_image_label = line_image_label.strip().split()
    image_id = line_image_label[0]
    boxes_nums = int(line_image_label[1])
    box = []
    label = []
    for i in range(boxes_nums):
        x = float(line_image_label[2 + 5 * i])
        y = float(line_image_label[3 + 5 * i])
        w = float(line_image_label[4 + 5 * i])
        h = float(line_image_label[5 + 5 * i])
        c = int(line_image_label[6 + 5 * i])
        if w <= 0 or h <= 0:
            continue
        box.append([x, y, x + w, y + h])
        label.append(c)
    return image_id, box, label


def read_lines_image_labels(filename):
    """
    :param filename:
    :return:
    """
    boxes_label_lists = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            image_id, box, label = read_line_image_label(line)
            boxes_label_lists.append([image_id, box, label])
    return boxes_label_lists


def split_letters_and_numbers(string, join=True):
    """
    切分字母和数字
    :param string:
    :param join:
    :return:
    """
    letters = []
    numbers = []
    for char in string:
        if char.isalpha():  # 判断字符是否是字母
            letters.append(char)
        elif char.isdigit():  # 判断字符是否是数字
            numbers.append(char)
    if join:
        letters = "".join(letters)
        numbers = "".join(numbers)
    return letters, numbers


def is_number(value):
    if re.match(r'^[-+]?[0-9]+(\.[0-9]+)?$', value):
        return True
    return False


def is_int(str):
    """
    判断是否为整数
    :param str:
    :return:
    """
    try:
        x = int(str)
        return isinstance(x, int)
    except ValueError:
        return False


def is_float(str):
    """
    判断是否为整数和小数
    :param str:
    :return:
    """
    try:
        x = float(str)
        return isinstance(x, float)
    except ValueError:
        return False


def list2str(content_list):
    """
    convert list to string
    :param content_list:
    :return:
    """
    content_str_list = []
    for line_list in content_list:
        line_str = " ".join('%s' % id for id in line_list)
        content_str_list.append(line_str)
    return content_str_list


def get_basename(file_list):
    """
    get files basename
    :param file_list:
    :return:
    """
    dest_list = []
    for file_path in file_list:
        basename = os.path.basename(file_path)
        dest_list.append(basename)
    return dest_list


def randam_select_images(image_list, nums, shuffle=True):
    """
    randam select nums images
    :param image_list:
    :param nums:
    :param shuffle:
    :return:
    """
    image_nums = len(image_list)
    if image_nums <= nums:
        return image_list
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
    out = image_list[:nums]
    return out


def remove_dir(dir):
    """
    remove directory
    :param dir:
    :return:
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)


def get_config_file(file_dir, prefix="*.yaml"):
    """
    获得config.yaml文件
    :param file_dir:
    :param prefix:
    :return:
    """
    if os.path.isfile(file_dir): file_dir = os.path.dirname(file_dir)
    files = get_prefix_files(file_dir, prefix)
    file = files[0] if len(files) > 0 else ""
    return file


def get_prefix_files(file_dir, prefix):
    """
    获得符合前缀条件所有文件
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = glob.glob(os.path.join(file_dir, prefix))
    return file_list


def remove_prefix_files(file_dir, prefix):
    """
    删除符合前缀条件所有文件
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = get_prefix_files(file_dir, prefix)
    for file in file_list:
        if os.path.isfile(file):
            remove_file(file)
        elif os.path.isdir(file):
            remove_dir(file)


get_files_with_prefix = get_prefix_files
remove_files_with_prefix = remove_prefix_files


def get_file_prefix_postfix(filename):
    """
    获得文件的前缀prefix和后缀postfix
    对于path/to/file.txt其前缀prefix='path/to/file'，后缀postfix='txt'
    :param filename:
    :return:
    """
    postfix = filename.split(".")[-1]
    prefix = filename[:-len(postfix) - 1]
    return prefix, postfix


def get_files_id(file_list):
    """
    :param file_list:
    :return:
    """
    image_idx = []
    for path in file_list:
        basename = os.path.basename(path)
        id = basename.split(".")[0]
        image_idx.append(id)
    return image_idx


def remove_file(path):
    """
    remove files
    :param path:
    :return:
    """
    if os.path.exists(path):
        os.remove(path)


def remove_file_list(file_list):
    """
    remove file list
    :param file_list:
    :return:
    """
    for file_path in file_list:
        remove_file(file_path)


def copy_dir_multi_thread(sync_source_root, sync_dest_dir, dataset, max_workers=1):
    """
    :param sync_source_dir:
    :param sync_dest_dir:
    :param dataset:
    :return:
    """

    def rsync_cmd(source_dir, dest_dir):
        cmd_line = "rsync -a {0} {1}".format(source_dir, dest_dir)
        # subprocess.call(cmd_line.split())
        subprocess.call(cmd_line)

    sync_dest_dir = sync_dest_dir.rstrip('/')

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_rsync = {}
        for source_dir in dataset:
            sync_source_dir = os.path.join(sync_source_root, source_dir.strip('/'))
            future_to_rsync[executor.submit(rsync_cmd, sync_source_dir, sync_dest_dir)] = source_dir

        for future in concurrent.futures.as_completed(future_to_rsync):
            source_dir = future_to_rsync[future]
            try:
                _ = future.result()
            except Exception as exc:
                print("%s copy data generated an exception: %s" % (source_dir, exc))
            else:
                print("%s copy data successful." % (source_dir,))


def copy_dir_delete(src, dst):
    """
    copy src directory to dst directory,will detete the dst same directory
    :param src:
    :param dst:
    :return:
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # time.sleep(3 / 1000.)


def copy_dir(src, dst, sub=False, exclude=[]):
    """ copy src-directory to dst-directory, will cover the same files"""
    if not os.path.exists(src):
        print("\nno src path:{}".format(src))
        return
    if sub: dst = os.path.join(dst, os.path.basename(src))
    for root, dirs, files in os.walk(src, topdown=False):
        isExclude = False
        if exclude:
            for p2 in exclude:
                p2 = p2[2:] if p2.startswith("./") else p2
                p1 = root[len(src) + 1:]
                isExclude = p1.startswith(p2)
                if isExclude: break
        if isExclude: continue
        dest_path = os.path.join(dst, os.path.relpath(root, src))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for filename in files:
            copy_file(
                os.path.join(root, filename),
                os.path.join(dest_path, filename)
            )


def move_dir(src, dst, sub=False):
    """ copy src-directory to dst-directory, will cover the same files"""
    if not os.path.exists(src):
        print("\nno src path:{}".format(src))
        return
    if sub: dst = os.path.join(dst, os.path.basename(src))
    for root, dirs, files in os.walk(src, topdown=False):
        dest_path = os.path.join(dst, os.path.relpath(root, src))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for filename in files:
            move_file(
                os.path.join(root, filename),
                os.path.join(dest_path, filename)
            )


def move_file(srcfile, dstfile):
    """ 移动文件或重命名"""
    if os.path.exists(srcfile) and os.path.isfile(srcfile):
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)
        # print("copy %s -> %s"%( srcfile,dstfile))
        # time.sleep(1 / 1000.)
    else:
        print("%s not exist!" % (srcfile))


def copy_file(srcfile, dstfile):
    """
    copy src file to dst file
    :param srcfile:
    :param dstfile:
    :return:
    """
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        # print("copy %s -> %s"%( srcfile,dstfile))
        # time.sleep(1 / 1000.)


def copy_file_to_dir(srcfile, des_dir):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)  # 创建路径
        dstfile = os.path.join(des_dir, fname)
        shutil.copyfile(srcfile, dstfile)  # 复制文件


def move_file_to_dir(srcfile, des_dir):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)  # 创建路径
        dstfile = os.path.join(des_dir, fname)
        # shutil.copyfile(srcfile, dstfile)  # 复制文件
        move_file(srcfile, dstfile)  # 复制文件


def copy_file_list(file_list, dst_dir):
    [copy_file_to_dir(file, dst_dir) for file in file_list]


def move_file_list(file_list, dst_dir):
    [move_file_to_dir(file, dst_dir) for file in file_list]


def merge_dir(src, dst, sub, merge_same):
    src_dir = os.path.join(src, sub)
    dst_dir = os.path.join(dst, sub)

    if not os.path.exists(src_dir):
        print("\nno src path:{}".format(src))
        return
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    elif not merge_same:
        t = get_time()
        dst_dir = os.path.join(dst, sub + "_{}".format(t))
        print("have save sub:{}".format(dst_dir))
    copy_dir(src_dir, dst_dir)


def create_dir(parent_dir, dir1=None, filename=None):
    """
    create directory
    :param parent_dir:
    :param dir1:
    :param filename:
    :return:
    """
    out_path = parent_dir
    if dir1:
        out_path = os.path.join(parent_dir, dir1)
    if not out_path: return out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if filename:
        out_path = os.path.join(out_path, filename)
    return out_path


def create_file_path(filename):
    """
    create file in path
    :param filename:
    :return:
    """
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    out_path = create_dir(dirname, dir1=None, filename=basename)
    return out_path


def get_sub_paths(input_dir):
    """
    当前路径下所有子目录
    :param input_dir:
    :return:
    """
    sub_list = []
    for root, dirs, files in os.walk(input_dir):
        sub_list = dirs
        break
    # print(root)   # 当前目录路径
    # print(dirs)   # 当前路径下所有子目录
    # print(files)  # 当前路径下所有非目录子文件
    sub_list.sort()
    return sub_list


def get_sub_list(file_list, dirname: str):
    """
    去除父路径,获得子路径:file_list = file_list - dirname
    :param file_list:
    :param parent:
    :return:
    """
    dirname = dirname[:-len(os.sep)] if dirname.endswith(os.sep) else dirname
    sub_list = []
    for i, f in enumerate(file_list):
        if dirname == f[:len(dirname)]:
            sub_list.append(f[len(dirname) + 1:])
    return sub_list


def get_train_test_files(file_dir, ratio=0.2, postfix=IMG_POSTFIX, subname="", shuffle=False, sub=False, save=True):
    """
    划分训练集和测试集
    :param file_dir:
    :param ratio: 若小于0，则表示test/train的比例；若大于0，则表示test的样本数，剩下的为train数目
    :param postfix:
    :param subname:
    :param shuffle:
    :param sub:
    :return:
    """
    file_list = get_files_lists(file_dir, postfix=postfix, subname=subname,
                                shuffle=shuffle, sub=sub)
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    nums = len(file_list)
    test_nums = ratio if ratio > 1 else int(len(file_list) * ratio)
    test, train = file_list[0:test_nums], file_list[test_nums:]
    train.sort()
    test.sort()
    if os.path.isdir(file_dir):
        out = file_dir
    elif os.path.isfile(file_dir):
        out = os.path.dirname(file_dir)
    else:
        out = file_dir
    if save:
        write_list_data(os.path.join(out, f"total-{nums}.txt"), file_list)
        write_list_data(os.path.join(out, f"train-{len(train)}.txt"), train)
        write_list_data(os.path.join(out, f"test-{len(test)}.txt"), test)
    print("total files: {}".format(nums))
    print("train files: {}".format(len(train)))
    print("test  files: {}".format(len(test)))
    return train, test


def get_all_files(file_dir):
    """获取file_dir目录下，所有文本路径，包括子目录文件"""
    file_list = []
    for walk in os.walk(file_dir):
        # paths = [os.path.join(walk[0], file).replace("\\", "/") for file in walk[2]]
        paths = [os.path.join(walk[0], file) for file in walk[2]]
        file_list.extend(paths)
    return file_list


def get_files_lists(file_dir, postfix=IMG_POSTFIX, subname="", shuffle=False, sub=False):
    """
    读取文件和列表: list,*.txt ,image path, directory
    :param file_dir: list,*.txt ,image path, directory
    :param subname: "JPEGImages"
    :param sub: 是否去除根路径
    :return:
    """
    if isinstance(file_dir, list):
        file_list = file_dir
    elif file_dir.endswith(".txt"):
        data_root = os.path.dirname(file_dir)
        file_list = read_data(file_dir, split=None)
        if subname: file_list = [os.path.join(data_root, subname, n) for n in file_list]
    elif os.path.isdir(file_dir):
        file_list = get_files_list(file_dir, prefix="", postfix=postfix)
    elif os.path.isfile(file_dir):
        file_list = [file_dir]
    else:
        file_list = [file_dir]
        # raise Exception("Error:{}".format(file_dir))
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    if sub: file_list = get_sub_list(file_list, dirname=file_dir)
    return file_list


def get_files_list(file_dir, prefix="", postfix=None, basename=False, sub=False):
    """
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录所有文件
    :param file_dir:
    :param prefix: 前缀
    :param postfix: 后缀
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :param sub: 是否去除根路径
    :return:
    """
    file_list = []
    paths = get_all_files(file_dir)
    if postfix is None:
        file_list = paths
    else:
        postfix = [p.split('.')[-1].lower() for p in postfix]
        prefix = prefix.lower()
        for file in paths:
            name = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = name.split('.')[-1].lower()
            prefix_name = name[:len(prefix)].lower()
            if prefix_name == prefix and postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    file_list = get_basename(file_list) if basename else file_list
    if sub: file_list = get_sub_list(file_list, dirname=file_dir)
    return file_list


def get_files_list_v2(file_dir, prefix="", postfix=None, basename=False):
    """
    获取file_dir目录下，所有文本路径，但不包括子目录文件
    :param file_dir:
    :param prefix: 前缀
    :param postfix: 后缀
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :return:
    """
    file_list = []
    if not postfix:
        file_list = glob.glob(os.path.join(file_dir, prefix + "*"))
    else:
        postfix = [postfix] if isinstance(postfix, str) else postfix
        for p in postfix:
            dir = os.path.join(file_dir, prefix + p)
            item = glob.glob(dir)
            file_list = file_list + item if item else file_list
    file_list.sort()
    file_list = get_basename(file_list) if basename else file_list
    return file_list


def get_images_list(file_dir, prefix="", postfix=IMG_POSTFIX, basename=False):
    """
    :param file_dir:
    :param prefix: 前缀
    :param postfix: 后缀
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :return:
    """
    return get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=basename)


def get_files_labels(file_dir, prefix="", postfix=IMG_POSTFIX, basename=False):
    '''
    获取files_dir路径下所有文件路径，以及labels,其中labels用子级文件名表示
    files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
    :param file_dir:
    :param prefix: 前缀
    :param postfix: 后缀
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :return:file_list所有文件的路径,label_list对应的labels
    '''
    file_list = get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=basename)
    sub_list = get_sub_list(file_list, file_dir)
    label_list = []
    for filePath in sub_list:
        label = filePath.split(os.sep)[0]
        label_list.append(label)
    return file_list, label_list


def save_file_list(file_dir, filename=None, prefix="", postfix=IMG_POSTFIX, only_id=True, shuffle=False, max_num=None):
    """
    保存文件列表
    :param file_dir: 文件路径
    :param filename: 输出文件列表
    :param prefix:   需要筛选的文件前缀
    :param postfix:  需要筛选的文件后缀
    :param only_id:  是否去除后缀，只保留ID
    :param shuffle:  是否打乱顺序
    :param max_num:  最大文件数
    :return:
    """
    if not filename: filename = os.path.join(os.path.dirname(file_dir), "file_list.txt")
    file_list = get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=False)
    file_list = get_sub_list(file_list, dirname=file_dir)
    if only_id:
        file_list = [str(f).split(".")[0] for f in file_list]
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    if max_num:
        max_num = min(max_num, len(file_list))
        file_list = file_list[0:max_num]
    write_list_data(filename, file_list)
    print("num files:{},out_path:{}".format(len(file_list), filename))
    return file_list


def decode_label(label_list, name_table):
    '''
    根据name_table解码label
    :param label_list:
    :param name_table:
    :return:
    '''
    name_list = []
    for label in label_list:
        name = name_table[label]
        name_list.append(name)
    return name_list


def encode_label(name_list, name_table, unknow=0):
    '''
    根据name_table，编码label
    :param name_list:
    :param name_table:
    :param unknow :未知的名称，默认label为0,一般在name_table中index=0是背景，未知的label也当做背景处理
    :return:
    '''
    label_list = []
    # name_table = {name_table[i]: i for i in range(len(name_table))}
    for name in name_list:
        if name in name_table:
            index = name_table.index(name)
        else:
            index = unknow
        label_list.append(index)
    return label_list


def list2dict(data):
    """
    convert list to dict
    :param data:
    :return:
    """
    data = {data[i]: i for i in range(len(data))}
    return data


def print_dict(dict_data, save_path):
    """
    print dict info
    :param dict_data:
    :param save_path:
    :return:
    """
    list_config = []
    for key in dict_data:
        info = "conf.{}={}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")


def get_pair_data(image_dir, pair_num=-1):
    """
    获得图片对数据
    :param image_dir:
    :param pair_num:-1 表示所有对
    :return:
    """
    max_nums = int(pair_num / 2)
    image_list = get_files_lists(image_dir)
    image_list = get_sub_list(image_list, dirname=image_dir)
    nums = len(image_list)
    print("have {} images and {} combinations".format(nums, nums * (nums - 1) / 2))
    pairs = []
    for paths in itertools.combinations(image_list, 2):
        file1, file2 = paths
        label1 = file1.split(os.sep)[0]
        label2 = file2.split(os.sep)[0]
        if label1 == label2:
            pairs.append([file1, file2, 1])
        else:
            pairs.append([file1, file2, 0])
    pairs = np.asarray(pairs)
    pairs = pairs[np.lexsort(pairs.T)]
    pair0 = pairs[pairs[:, -1] == "0", :]
    pair1 = pairs[pairs[:, -1] == "1", :]
    nums1 = len(pair1)
    nums0 = len(pair0)
    if pair_num < 0: max_nums = nums1
    if max_nums > nums1:
        raise Exception("pair_nums({}) must be less than num_pair1({})".format(max_nums, nums1))
    index_0 = np.random.permutation(nums0)[:max_nums]  # 打乱后的行号
    index_1 = np.random.permutation(nums1)[:max_nums]  # 打乱后的行号
    pair0 = pair0[index_0, :]  # 获取打乱后的训练数据
    pair1 = pair1[index_1, :]  # 获取打乱后的训练数据
    pairs = np.concatenate([pair0, pair1], axis=0).tolist()
    print("have {} pairs，pair0 nums:{}，pair1 nums:{}".format(len(pairs), len(pair0), len(pair1)))
    return pairs


def get_pair_files(data_root, out_root=None, image_sub="", label_sub="",
                   postfix=IMG_POSTFIX, label_postfix="txt", shuffle=False):
    """
    获得同目录下一对文件
    :param data_root: 根目录
    :param out_root:
    :param image_sub:
    :param label_sub:
    :param label_postfix: label文件后缀，如txt,png,json等
    :return:
    """
    image_dir = os.path.join(data_root, image_sub)
    label_dir = os.path.join(data_root, label_sub)
    if out_root: create_dir(out_root)
    file_list = get_files_lists(file_dir=data_root, postfix=postfix, subname="", shuffle=shuffle, sub=False)
    pair_list = []
    for i, image_name in tqdm(enumerate(file_list)):
        postfix = image_name.split(".")[-1]
        lable_name = image_name.replace(f".{postfix}", f".{label_postfix}")
        image_file = os.path.join(image_dir, image_name)
        lable_file = os.path.join(label_dir, lable_name)
        if os.path.exists(image_file) and os.path.exists(lable_file):
            image_file, lable_file = get_sub_list([image_file, lable_file], dirname=data_root)
            pair_list.append([image_file, lable_file])
    if out_root:
        filename = os.path.join(out_root, "file_list.txt")
        write_data(filename, pair_list, split=",", mode='w')
    return pair_list


def read_pair_data(filename, split=True):
    """
    read pair data,data:[image1.jpg image2.jpg 0]
    :param filename:
    :param split:
    :return:
    """
    pair_list = read_data(filename)
    if split:
        pair_list = np.asarray(pair_list)
        pair1 = pair_list[:, :1].reshape(-1)
        pair2 = pair_list[:, 1:2].reshape(-1)
        # convert to 0/1
        issames_data = np.asarray(pair_list[:, 2:3].reshape(-1), dtype=np.int)
        issames_data = np.where(issames_data > 0, 1, 0)
        pair1 = pair1.tolist()
        pair2 = pair2.tolist()
        issames_data = issames_data.tolist()
        return pair1, pair2, issames_data
    return pair_list


def check_files(files_list, sizeTh=1 * 1024, isRemove=False):
    """
    去除不存的文件和文件过小的文件列表
    :param files_list:
    :param sizeTh: 文件大小阈值,单位：字节B，默认1000B ,33049513/1024/1024=33.0MB
    :param isRemove: 是否在硬盘上删除被损坏的原文件
    :return:
    """
    i = 0
    while i < len(files_list):
        path = files_list[i]
        # 判断文件是否存在
        if not (os.path.exists(path)):
            print(" non-existent file:{}".format(path))
            files_list.pop(i)
            continue
        # 判断文件是否为空
        f_size = os.path.getsize(path)
        if f_size < sizeTh:
            print(" empty file:{}".format(path))
            if isRemove:
                os.remove(path)
                print(" info:----------------remove image_dict:{}".format(path))
            files_list.pop(i)
            continue
        i += 1
    return files_list


def get_set_inter_union_diff(set1, set2):
    """
    intersection(交集),union(并集),difference(差集)
    :return:
    """
    # 两个列表的差集
    difference = list(set(set1) ^ set(set2))
    # 获取两个list 的交集
    intersection = list(set(set1) & set(set2))
    # 获取两个list 的并集
    union = list(set(set1) | set(set2))
    dset1 = list(set(set1) - set(set2))  # 去除set1中含有set2的元素
    dset2 = list(set(set2) - set(set1))  # 去除set2中含有set1的元素
    return intersection, union, difference


def get_loacl_eth2():
    '''
    想要获取linux设备网卡接口，并用列表进行保存
    :return:
    '''
    eth_list = []
    os.system("ls -l /sys/class/net/ | grep -v virtual | sed '1d' | awk 'BEGIN {FS=\"/\"} {print $NF}' > eth.yaml")
    try:
        with open('./eth.yaml', "rb") as f:
            for line in f.readlines():
                line = line.strip()
                eth_list.append(line.lower())
    except Exception as e:
        print(e)
        eth_list = []
    return eth_list


def get_loacl_eth():
    '''
    想要获取linux设备网卡接口，并用列表进行保存
    :return:
    '''
    eth_list = []
    cmd = "ls -l /sys/class/net/ | grep -v virtual | sed '1d' | awk 'BEGIN {FS=\"/\"} {print $NF}'"
    try:
        with os.popen(cmd) as f:
            for line in f.readlines():
                line = line.strip()
                eth_list.append(line.lower())
    except Exception as e:
        print(e, "can not found eth,will set default eth is:eth0")
        eth_list = ["eth0"]
    if not eth_list:
        eth_list = ["eth0"]
    return eth_list


def merge_files(files_list):
    """
    合并文件列表
    :return:
    """
    content_list = []
    for file in files_list:
        data = read_data(file)

    return content_list


def multi_thread_task(content_list, func, num_processes=4, remove_bad=False, Async=True, **kwargs):
    """
    多线程处理content_list的数据
    Usage:
        def task_fun(item, save_root):
            '''
            :param item: 对应content_list的每一项item
            :param save_root: 对应kwargs
            :return:
            '''
            pass
        multi_thread_task(content_list,
                          func=task_fun,
                          num_processes=num_processes,
                          remove_bad=remove_bad,
                          Async=Async,
                          save_root=save_root)
    =====================================================
    :param content_list: content_list
    :param func: func：task function
    :param num_processes: 开启线程个数
    :param remove_bad: 是否去除下载失败的数据
    :param Async:是否异步
    :param kwargs:需要传递给func的相关参数
    :return: 返回图片的存储地址列表
    """
    from multiprocessing.pool import ThreadPool
    # 开启多线程
    pool = ThreadPool(processes=num_processes)
    thread_list = []
    for item in content_list:
        if Async:
            out = pool.apply_async(func=func, args=(item,), kwds=kwargs)  # 异步
        else:
            out = pool.apply(func=func, args=(item,), kwds=kwargs)  # 同步
        thread_list.append(out)

    pool.close()
    pool.join()
    # 获取输出结果
    dst_content_list = []
    if Async:
        for p in thread_list:
            image = p.get()  # get会阻塞
            dst_content_list.append(image)
    else:
        dst_content_list = thread_list
    if remove_bad:
        dst_content_list = [i for i in dst_content_list if i is not None]
    return dst_content_list


def save_pickle(obj, file):
    with open(file, 'wb') as f: pickle.dump(obj, f)


def load_pickle(file):
    with open(file, 'rb') as f: obj = pickle.load(f)
    return obj


def copy_move_file_dir(src, dst, postfix=None, sub_names=None, max_nums=None, shuffle=True, move=False):
    """
    复制/移动文件夹
    :param src: 原始目录
    :param dst: 输出目录
    :param sub_names: 需要复制/移动的子文件夹名称，默认全部
    :param max_nums: 最大移动数量，默认不限制
    :param shuffle:
    :param move: True表示移动，False表示复制
    :return: out_list 返回复制/移动后，out_dir的文件列表
    """
    file_list = get_files_list(src, prefix="", postfix=postfix, basename=False)
    file_list = get_sub_list(file_list, src)
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
        random.shuffle(file_list)
    if max_nums: file_list = file_list[:min(max_nums, len(file_list))]
    for file_name in tqdm(file_list):
        sub = os.path.dirname(file_name)
        if sub_names and (sub not in sub_names): continue
        src_file = os.path.join(src, file_name)
        out_file = os.path.join(dst, file_name)
        if move:
            move_file(src_file, out_file)
        else:
            copy_file(src_file, out_file)
    out_list = get_files_list(dst, prefix="", postfix=postfix, basename=False)
    return out_list


def copy_move_dir_dir(src, dst, postfix=None, sub_names=None, per_nums=None, shuffle=True, move=False):
    """
    复制/移动文件夹
    :param src: 原始目录
    :param out_dir_: 输出目录
    :param sub_names: 需要复制/移动的子文件夹名称，默认全部
    :param per_nums: 每个子文件夹最大移动数量，默认全部
    :param shuffle:
    :param move: True表示移动，False表示复制
    :return: out_list 返回复制/移动后，out_dir的文件列表
    """
    sub_list = get_sub_paths(src)
    for sub in tqdm(sub_list):
        if sub_names and (sub not in sub_names): continue
        src_dir_ = os.path.join(src, sub)
        out_dir_ = os.path.join(dst, sub)
        if per_nums is None:
            if move:
                move_dir(src_dir_, out_dir_)
            else:
                copy_dir(src_dir_, out_dir_)
        else:
            copy_move_file_dir(src_dir_, out_dir_, postfix=postfix, sub_names=postfix,
                               max_nums=per_nums, shuffle=shuffle, move=move)
    out_list = get_files_list(dst, prefix="", postfix=postfix, basename=False)
    return out_list


def get_voc_file_list(voc_root,
                      image_dir="",
                      annos_dir="",
                      prefix="",
                      postfix=IMG_POSTFIX,
                      only_id=False,
                      check=True,
                      shuffle=False,
                      max_num=None):
    """
    获得VOC数据集的文件列表，并剔除无效的文件
    :param voc_root:
    :param prefix:
    :param postfix:
    :param only_id:
    :param check: 检测xml和图片是否存在
    :param shuffle:
    :param max_num:
    :return:
    """
    image_dir = image_dir if image_dir else os.path.join(voc_root, "JPEGImages")
    annos_dir = annos_dir if annos_dir else os.path.join(voc_root, "Annotations")
    filename = os.path.join(os.path.dirname(image_dir), "file_list.txt")
    file_list = get_files_list(image_dir, prefix=prefix, postfix=postfix, basename=False)
    file_list = get_sub_list(file_list, dirname=image_dir)
    if check:
        xmls_list = get_files_list(annos_dir, postfix=["*.xml"], basename=True)
        print("xml file:{},image file:{}".format(len(xmls_list), len(file_list)))
        xmls_ids = [str(f).split(".")[0] for f in xmls_list]
        file_list = [f for f in file_list if str(f).split(".")[0] in xmls_ids]
    if only_id:
        file_list = [str(f).split(".")[0] for f in file_list]
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    if max_num:
        max_num = min(max_num, len(file_list))
        file_list = file_list[0:max_num]
    write_list_data(filename, file_list)
    print("num files:{},out_path:{}".format(len(file_list), filename))


def copy_move_voc_dataset(data_file, data_root=None, out_root=None, file_map={}, move=False):
    """
    :param data_file: voc file.txt
    :param data_root: voc dataset root path
    :param out_root:  new output root path
    :param file_map: {"Annotations": "xml", "json": "json", "JPEGImages": None}
    :param move: move or copy
    :return:
    """
    if not data_root: data_root = os.path.dirname(data_file)
    files = read_data(data_file, split=None)
    for name in tqdm(files):
        idx, pos = name.split(".")
        for sub, p in file_map.items():
            n = name.replace(f".{pos}", f".{p}") if p else name
            path = os.path.join(data_root, sub, n)
            if out_root and os.path.exists(path):
                dest = os.path.join(out_root, sub, n)
                if move:
                    move_file(path, dest)
                else:
                    copy_file(path, dest)
            else:
                print(f"no file:{path}")
    return files


if __name__ == '__main__':
    dir = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval/unknown"
    file_list, label_list = get_files_labels(dir)
    print(label_list)
