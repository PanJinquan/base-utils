# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-11-23 11:54:52
    @Brief  :
"""


def find_synonym(synonym, key):
    """
    查找同义词
    :param synonym:
    :param key:
    :return:
    """
    group = -1
    for i in range(len(synonym)):
        # 判断keyword是否是同义词中
        if key in synonym[i]:
            group = i
            break
    return group


def get_keyword_table(keyword: dict, synonym: list):
    key_table = {}
    for word, w in keyword.items():
        index = find_synonym(synonym, key=word)
        group = synonym[index] if index >= 0 else [word]
        for key in group:
            key_table[key] = {"synonym": group, "weight": w, "index": index}
    return key_table


def split_keyword(eva_text: str, keyword: list):
    for key in keyword:
        eva_text = eva_text.replace(key, f"#{key}#")
    eva_text = [t for t in eva_text.split("#") if t]
    return eva_text


if __name__ == "__main__":
    keyword = {'程序猿': 0.5, '工作服': 0.3, '安全帽': 0.1}
    synonym = [["很好"], ['程序员', '程序猿', ], ['工作服', '衣服']]
    # key_table = get_keyword_table(keyword, synonym)
    eva_text = '好的，我是一位IT程序猿，喜欢穿工作服工作服的服装'
    key_table = split_keyword(eva_text, list(keyword.keys()))
    print(key_table)
