# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-02 16:03:30
    @Brief  :
"""

from pybaseutils import file_utils, json_utils


def startswith(line, keys: list):
    for k in keys:
        if line.startswith(k): return True
    return False


def read_exam_info(content, exam_info={}, exam_nums=1, title="单选题", start=["一、单选题"], end=["二、"]):
    """
    :param content:
    :return:
    """
    item_list = []
    key = ""
    nums = len(content)
    record = False
    item_info = json_utils.get_value(exam_info, [title, "考题列表"], default={})
    for i, line in enumerate(content):
        line = line.strip()
        if startswith(line, start):  record = True
        if startswith(line, end):  record = False
        split = line.split("、")
        if file_utils.is_int(split[0]):  # 通过序列判断每道题
            if key and item_list:
                count = json_utils.get_value(item_info, [key, "出现次数"], default=0) + 1
                item_info[key] = {"考题": item_list, "出现次数": count}
            line = line[len(split[0] + "、"):]
            key = line
            item_list = []
        if record: item_list.append(line)
        if i == nums - 1:
            if key and item_list:
                count = json_utils.get_value(item_info, [key, "出现次数"], default=0) + 1
                item_info[key] = {"考题": item_list, "出现次数": count}

    item_info = sorted(item_info.items(), key=lambda x: x[1]["出现次数"], reverse=True)
    item_info = {k: v for k, v in item_info}
    exam_info[title] = {"考题列表": item_info, "模拟考试次数": exam_nums}
    return exam_info


def save_exam_info(data_root, exam_info):
    nums = "一二三四五六七八九十"
    tmp = "tmp.txt"
    num = 0
    for title, info in exam_info.items():
        item_info = info["考题列表"]
        exam_nums = info["模拟考试次数"]
        file = file_utils.WriterTXT(tmp)
        file.write_line("{}、".format(nums[num]) + title)
        num += 1
        ids = 0
        print("{}:{}".format(title, len(item_info)))
        for item, value in item_info.items():
            content = value['考题']
            count = value['出现次数']
            ids += 1
            content[0] = "{}、".format(ids) + content[0]
            # content[-1] = content[-1] + "（模拟考试{}次,出现{}次,出现概率:{}%）".format(exam_nums, count, count*100 // exam_nums) + "\n"
            # content[-1] = content[-1] + "（模拟考试出现{}次）".format(count) + "\n"
            file.write_line_list(content)

        filename = data_root + "({})-{}-考试题库和答案.txt".format(len(item_info), title)
        file_utils.copy_file(tmp, filename)


if __name__ == '__main__':
    # data_root = "/media/PKing/新加卷/个人文件/医学/中医规培-公共科目-方剂学-1603"
    # data_root = "/media/PKing/新加卷/个人文件/医学/中医规培-公共科目-内经-502"
    data_root = "/media/PKing/新加卷/个人文件/医学/中医规培-公共科目-温病学-1058"
    data_root = "/media/PKing/新加卷/个人文件/医学/中药1"
    text_list = file_utils.get_files_lists(data_root, postfix=["*.txt"])
    exam_info = {}
    end = ["版权所有", "二、"]
    for file in text_list:
        print("process:{}".format(file))
        content = file_utils.read_data(file, split=None, convertNum=False)
        exam_info = read_exam_info(content, exam_info, exam_nums=len(text_list), title="单选题", start=["一、单选题"], end=end)
    save_exam_info(data_root, exam_info)
