# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import random
import numpy as np
import math
import time


class ResampleExample(object):
    """样本均衡，重采样DataResampler的使用方法"""

    def __init__(self, item_list, label_index, shuffle=True, disp=False):
        """
        :param item_list: item_list=[item_0,item_1,...,item_n],
                          item_n= [path/to/image,label]
        :param label_index: label在item_n的index
        :param disp: 是否打印log信息
        """
        self.disp = disp
        self.shuffle = shuffle
        self.item_list = item_list
        self.resampler = DataResample(self.item_list,
                                      label_index=label_index,
                                      shuffle=self.shuffle,
                                      disp=self.disp)

    def __len__(self):
        print("start resampler, data will shuffle and update")
        # 更新resampler，实现每个epoch重新采样，避免样本数比较多的类别，没有加入训练
        self.item_list = self.resampler.update(self.shuffle)
        return len(self.item_list)

    def __getitem__(self, index):
        data_info = self.item_list[index]
        return data_info


class DataResample(object):
    """样本均衡，重采样的方法"""

    def __init__(self, item_list=[], class_name=None, label_index=1, interval=0, balance="mean", shuffle=True,
                 disp=False, **kwargs):
        """
        Usage:
        参考：ResampleExample例子的使用方法
        :param item_list:
        :param class_name:
        :param label_index:
        :param interval: 重采样时间间隔(秒)，低于该时间的不进行重采集，避免频繁采样
        :param balance:实现样本均衡策略,均衡力度：mean > log > sqrt > y
                        "y": 每个label样本数跟原来一样
                        "sqrt": 每个label样本取sqrt数，实现样本均衡
                        "log": 每个label样本取log数，实现样本均衡
                        "mean": 每个label样本取样本平均数，每个label的个数一样
        """
        self.tag = self.__class__.__name__
        self.src_item_list = item_list
        self.class_name = class_name
        self.label_index = label_index
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.balance = balance
        self.shuffle = shuffle
        self.disp = disp
        self.t0 = time.time()
        self.first_time = True
        self.interval = interval  # TODO 重采样间隔，低于该时间的不进行重采集，避免频繁采样
        self.src_class_info = self.get_class_info(self.src_item_list, label_index)  # 原始数据样本分布
        self.src_class_count = {k: len(v) for k, v in self.src_class_info.items()}
        self.dst_class_count = self.get_balance_nums(self.src_class_count, self.balance)
        self.dst_class_info = {}
        self.item_list = []
        self.item_list = self.update(shuffle=self.shuffle)
        self.class_weight = self.get_class_weight(self.src_class_count)
        self.log("{:15s} balance        :{}".format(self.tag, self.balance))
        self.log("{:15s} interval       :{}".format(self.tag, self.interval))
        self.log("{:15s} interval       :{}".format(self.tag, self.interval))

    def __len__(self):
        self.update(shuffle=self.shuffle)
        return len(self.item_list)

    def update(self, shuffle=False):
        self.t1 = time.time()  # seconds
        dt = (self.t1 - self.t0)
        if not self.item_list or dt > self.interval:
            seed = int(self.t1) # TODO 修复BUG，由于其他地方有设置seed，导致每次shuffle结果都一致
            self.log(f"{self.tag:15s} resample dataset,seed={seed}")
            self.item_list = self.get_resample_data(shuffle=shuffle, seed=seed)
            self.t0 = self.t1
        return self.item_list

    def get_resample_data(self, shuffle=True, seed=2025):
        """
        获得重采样的数据
        :param shuffle:
        :param seed:
        :return:
        """
        random.seed(seed)  # TODO 修复BUG，由于其他地方有设置seed，导致每次shuffle结果都一致
        if self.disp or self.first_time:  # 统计每个类别的个数
            self.print_class_info("src_class_info", self.src_class_info, class_name=self.class_name)
        out_list = []
        for name, per_class_list in self.src_class_info.items():
            nums = self.dst_class_count[name]
            data = self.get_sampler(per_class_list, nums, shuffle=shuffle)
            out_list += data
        if shuffle:
            random.shuffle(out_list)
        if self.disp or self.first_time:  # 统计每个类别的个数
            self.dst_class_info = self.get_class_info(out_list, self.label_index)  # 原始数据样本分布
            self.print_class_info("dst_class_info", self.dst_class_info, class_name=self.class_name)
        self.first_time = False
        return out_list

    def print_class_info(self, title: str, class_info: dict, class_name=None):
        if class_name:
            info = {class_name[k]: len(v) for k, v in class_info.items()}
        else:
            info = {k: len(v) for k, v in class_info.items()}
        self.log("{:15s} {}: {}, total: {}".format(self.tag, title, info, sum(info.values())))

    def get_balance_nums(self, class_count: dict, balance):
        """
        获得平衡后，每个样本的数目
        :param class_count:
        :param balance:
        :return:
        """
        class_name = list(class_count.keys())
        num_samples = sum(class_count.values())  # 总样本数目
        if balance == "mean":
            mean_samples = num_samples * 1.0 / len(class_name)  # 平均样本数
            balance_nums = {name: mean_samples for name, c in class_count.items()}
        elif balance == "log":
            # Fix Bug:c=1
            balance_nums = {name: np.log(c + 1) for name, c in class_count.items()}
        elif balance == "sqrt":
            # Fix Bug:c=0
            balance_nums = {name: np.sqrt(c + 1) for name, c in class_count.items()}
        elif balance == "y":
            balance_nums = {name: c for name, c in class_count.items()}
        else:
            raise Exception("Error:{}".format(balance))
        sum_balance = sum(balance_nums.values())
        balance_nums = {name: math.ceil(c / sum_balance * num_samples) for name, c in balance_nums.items()}
        return balance_nums

    def get_sampler(self, item_list, nums, shuffle=True):
        """
        提取nums个数，不足nums个时，会进行填充
        :param item_list: 输入样本列表
        :param nums: 需要提取的样本数目
        :param shuffle: 是否随机提取样本
        :return:
        """
        item_nums = len(item_list)
        if nums > item_nums:
            item_list = item_list * math.ceil(nums / item_nums)
        if shuffle:
            random.shuffle(item_list)
        out_list = item_list[:nums]
        return out_list

    @staticmethod
    def get_label_list(item_list, label_index):
        labels_list = []
        for item in item_list:
            label = item[label_index]
            labels_list.append(label)
        return labels_list

    @staticmethod
    def get_class_info(item_list, label_index):
        """
        获得每一类的样本
        :param item_list:
        :return:
        """
        class_item_dict = {}
        for item in item_list:
            label = item[label_index]
            try:
                # if label in class_item_dict: # 比较慢，相当于需要查询label是否存在
                class_item_dict[label] += [item]
            except Exception as e:
                class_item_dict[label] = [item]
        return class_item_dict

    @staticmethod
    def get_class_count(item_list, label_index):
        """
        统计每个类别的个数
        :param item_list:
        :param label_index: label在item中的序号
        :return:
        """
        class_count = {}
        for item in item_list:
            label = item[label_index]
            try:
                # if label in class_count:  # 比较慢，相当于需要查询label是否存在
                class_count[label] += 1
            except Exception as e:
                class_count[label] = 1
        from pybaseutils import json_utils
        class_count = json_utils.dict_sort(class_count, use_key=True)
        return class_count

    @staticmethod
    def get_class_weight(class_count: dict):
        """
        计算每个label的权重，类别越少，权重越大
        :param class_count:
        :return:
        """
        n_samples = sum(list(class_count.values()))
        class_weight = {}
        for cls, num in class_count.items():
            score = n_samples / num
            class_weight[cls] = score
        return class_weight


class DataLabelResample(object):
    """样本均衡，重采样的方法"""

    def __init__(self, data, label, balance="mean", shuffle=True, disp=False):
        """
        Usage:
        参考：ResampleExample例子的使用方法
        :param item_list:
        :param label_index:
        :param balance:实现样本均衡策略,均衡力度：mean > log > sqrt > y
                        "y": 每个label样本数跟原来一样
                        "sqrt": 每个label样本取sqrt数，实现样本均衡
                        "log": 每个label样本取log数，实现样本均衡
                        "mean": 每个label样本取样本平均数，每个label的个数一样
        """
        self.shuffle = shuffle
        self.item_list = list(zip(data, label))
        self.resample = DataResample(item_list=self.item_list, label_index=1, balance=balance,
                                     shuffle=shuffle, disp=disp)

    def __len__(self):
        data, label = self.update(shuffle=self.shuffle)
        return len(data)

    def update(self, shuffle=False):
        item_list = self.resample.update(shuffle=shuffle)
        data = [item[0] for item in item_list]
        label = [item[1] for item in item_list]
        return data, label


def get_class_count(item_list, label_index):
    """
    统计每个类别的个数
    :param item_list: List[list,list]
    :param label_index: label在item中的序号
    :return:
    """
    class_count = {}
    for item in item_list:
        label = item[label_index]
        try:
            # if label in class_count:  # 比较慢，相当于需要查询label是否存在
            class_count[label] += 1
        except Exception as e:
            class_count[label] = 1
    from pybaseutils import json_utils
    class_count = json_utils.dict_sort(class_count, use_key=True)
    return class_count


def get_label_count(label):
    """
    file_utils.get_count_nums()统计列表元素的出现次数，然后找到出现次数最多的元素
    统计每个类别的个数
    :param label: [int,int,...]
    :return:
    """
    count = np.bincount(label).tolist()
    label = list(range(max(count)))
    count_info = {l: c for l, c in zip(label, count) if c > 0}
    return count_info


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader, Sampler
    from basetrainer.utils import torch_tools

    torch_tools.set_env_random_seed()
    label0 = [["0.1.jpg", 0], ["0.2.jpg", 0], ["0.3.jpg", 0]]
    label1 = [["1.jpg", 1]] * 5
    label2 = [["2.0.jpg", 2], ["2.1.jpg", 2], ["2.2.jpg", 2], ["2.3.jpg", 2], ["2.4.jpg", 2],
              ["2.5.jpg", 2], ["2.6.jpg", 2], ["2.7.jpg", 2], ["2.8.jpg", 2]]
    label3 = [["3.1.jpg", 3], ["3.2.jpg", 3], ["3.3.jpg", 3]]
    item_list = label0 + label1 + label2 + label3
    item_list = item_list * 1
    item_list = [{"file": item[0], "label": item[1]} for item in item_list]
    print("have item_list:{}".format(len(item_list)))
    dataset_train = ResampleExample(item_list=item_list, label_index="label", shuffle=True, disp=True)
    # dataset_train = ResampleExample(item_list=item_list, label_index=1, disp=False)
    # dataset_train = ResampleExample(item_list=item_list, label_index=1, disp=False)
    batch_size = 20
    dataloader = DataLoader(dataset_train, batch_size, num_workers=0)
    epochs = 2
    for epoch in range(epochs):
        print("{}===".format(epoch) * 10)
        for step, data in enumerate(tqdm(dataloader)):
            print(step, data)
