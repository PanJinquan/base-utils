# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  : https://www.pidancode.com/a/16814938447660138.html
# --------------------------------------------------------
"""

import re
import difflib



def find_match_text_index(text: str, pattern: str):
    """
    在text中，查找符合条件内容的所有index
        .  匹配任意单个字符(除换行符)
        ^  匹配字符串开头
        $  匹配字符串结尾
        *  前一个字符0次或多次
        +  前一个字符1次或多次
        ?  前一个字符0次或1次
        {m} 前一个字符m次
        {m,n} 前一个字符m到n次
        [] 字符集，匹配其中任意一个字符
        | 或，匹配左边或右边
        () 分组
    :param text:
    :param pattern:
    :return: index, 符合条件内容的所有index,如果要获得匹配的文本则 match = [text[i:j] for (i, j) in index]
    """
    index = []
    if not pattern:  return index  # 如果 str2 是空字符串，直接返回空列表
    res = re.finditer(pattern, text)
    for m in res:
        index.append([m.start(), m.end()])
    return index



def find_match_text(text: str, pattern: str):
    """
    使用通配符，在text中，查找符合条件内容
    :param text: 输入长字符串
    :param pattern: 需要匹配的子串
    :return:
    """
    if "*" in pattern:
        pattern = pattern.replace('*', '.*')  # 将通配符转换为正则表达式
        res = re.findall(pattern, text)  # 使用正则表达式匹配子串
    else:
        res = [text] if text == pattern else []
    return res


def find_match_texts(texts: list, pattern: list, org=True):
    """
    使用通配符，在texts列表中，查找符合条件内容
    :param texts: 输入List,包含多个长字符串
    :param pattern: 需要匹配的子串
    :param org:  True,返回匹配成功的原始字符串，False 返回匹配成功的子串
    :return:
    """
    if isinstance(pattern, str): pattern = [pattern]
    out = []
    for text in texts:
        for sub in pattern:
            res = find_match_text(text, sub)
            if res:
                out += [text] if org else res
    out = list(set(out))  # 去重复
    return out


def get_chinese_text(text: str, repl=""):
    """
    仅保留中文字符
    :param text:
    :param repl:
    :return:
    """
    text = re.sub(r'[^\u4e00-\u9fff]', repl, text)
    text = text.strip()
    return text


def rep_text(text, words=[], repl="", unique=False):
    """
    :param text:  输入文本
    :param words: 原始字词
    :param repl:  替换后的字词
    :param unique: 是否去除重复字词
    :return:
    """
    for w in words:
        text = text.replace(w, repl)
    text = text.strip()
    if not unique: return text
    res = ""
    for c in text:
        if c not in repl:
            res += c
        elif not res.endswith(c):
            res += c
    return res


def del_ignore_words(text: str, ignore_words=[], repl=""):
    """
    去除忽略词
    :param text:
    :param ignore_words:
    :return:
    """
    for w in ignore_words:
        text = text.replace(w, repl)
    text = text.strip()
    return text


def del_punctuation(text: str, repl=""):
    """
    去除标点符号
    :param text:
    :return:
    """
    # text = re.sub('[^\w\s]', repl, text)# 会把小数点等也去掉
    # text = re.sub('[,，。？?；;:：#￥$&*！!]', repl, text)# 会把冒号去除
    text = re.sub('[,，。？?；;#￥$&*！!]', repl, text)
    text = text.strip()
    return text


def get_synonym_texts(text: str or list[str], synonym=[]):
    """
    :param text: 输入文本，str or list[str]
    :param synonym: 同义词列表,如[["开心", "高兴", "愉快"], ["今天", "当天"]]
    :return: 返回同义词文本列表
    """

    def create_synonym(inp_texts: str or list[str], synonym: list):
        out_text = set()
        if not inp_texts: return out_text
        if isinstance(inp_texts, str): inp_texts = [inp_texts]
        for w1 in synonym:
            for w2 in synonym:
                text = [t.replace(w1, w2) for t in inp_texts]
                if text: out_text.update(text)
        return out_text

    out = [text]
    for i in range(len(synonym)):
        out = create_synonym(out, synonym=synonym[i])
    return out


def get_text_matching(ref_text, eva_text, min_size=1, ignore=[]):
    """
    进行文本匹配,并返回eva_text匹配文本的index和子段
    注意,交换匹配(A,B)和(B,A)，匹配内容可能不一致的
    m1=get_text_matching(text1, text2, min_size=1, ignore=[]) # text1 = 'ABCE'，text2 = 'ACDB'
    m2=get_text_matching(text2, text1, min_size=1, ignore=[])
    :param ref_text:
    :param eva_text:
    :param keyword: 关键词
    :param min_size:
    :return:
    """
    matches = difflib.SequenceMatcher(None, eva_text, ref_text).get_matching_blocks()
    result = []
    for match in matches:
        if match.size >= min_size:
            text = eva_text[match.a:match.a + match.size]
            if text in ignore: continue
            result.append({"text": text, "index": match.a, "offset": match.size})
    return result


def data_upper(data):
    """
    将大写大小
    """
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = data_upper(data[i])
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = data_upper(v)
    return data


def get_text_similarity(str1, str2):
    """
    计算文本相似度
    :param str1:
    :param str2:
    :return: simi 相似度
             dist 编辑距离
    """
    import editdistance
    size = max(len(str1), len(str2))
    dist = editdistance.eval(str1, str2)
    simi = 0.0
    if size > 0: simi = 1. - dist / size
    return simi, dist


def get_standard_text(text, ignore_words):
    """
    获得标准文本，用于测评
    :param text:
    :return:
    """
    text = del_ignore_words(text, ignore_words=ignore_words, repl=" ")  # 去除忽略词
    text = del_punctuation(text, repl=" ")  # 去除标点符号
    text = rep_text(text, words=[" "], repl="，", unique=True)  # 将标点符号统一转为，
    text = text.upper()
    return text


def insert_substring(string: str, index: int, sub: str):
    "在string中插入子串"
    return string[:index] + sub + string[index:]


def find_substring(string: str, sub=[]):
    """
    在string中查找多个子串(首个)
    :param string: 字符串
    :param sub: 匹配的子串列表[]
    :return: [(index,sub),...,]
    """
    if isinstance(sub, str): sub = []
    r = []
    for s in sub:
        i = string.find(s)
        if i != -1: r.append((i, s))
    return r


def find_substring_all(string: str, sub: str):
    """
    在string中查找所有子串
    :param string: 字符串
    :param sub: 匹配的子串列表[]
    :return: [(index,sub),...,]
    """
    r = []
    i = string.find(sub)
    while i != -1:
        r.append(i)
        i = string.find(sub, i + 1)  # 从下一个位置继续查找
    return r


def insert_string(text, index: int, sub: str):
    """
    在字符串指定位置插入子串
    :param text:
    :param index:
    :param sub:
    :return:
    """
    if index < 0: index = len(text) + index + 1
    if 0 <= index <= len(text):
        text = text[:index] + sub + text[index:]
    return text


def find_numbers(text):
    """
    提取字符串中的所有数字（包括小数和负号）
    -?  匹配可选的负号
    \d+ 匹配一个或多个数字
    \.? 匹配可选的小数点
    \d* 匹配零个或多个数字（小数部分）
    :param text:  要提取数字的字符串字
    :return: 所有数字的列表
    """
    out = re.findall(r'-?\d+\.?\d*', text)
    return out


def find_digits(text):
    """
    提取字符串中的所有数字(仅数字，不包括小数点和负号）
    \d 匹配数字字符（0-9）
    +  表示匹配一个或多个连续的数字字符
    :param text:  要提取数字的字符串字
    :return: 所有数字的列表
    """
    out = re.findall(r'\d+', text)
    return out


if __name__ == "__main__":
    text1 = 'ABCE'
    text2 = 'ACDB'
    print(get_text_matching(text1, text2, min_size=1, ignore=[]))
    print(get_text_matching(text2, text1, min_size=1, ignore=[]))
