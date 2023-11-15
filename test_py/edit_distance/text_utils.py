# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-11-08 14:53:44
    @Brief  : https://www.pidancode.com/a/16814938447660138.html
"""
import re
import difflib
import editdistance

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
    text = re.sub('[^\w\s]', repl, text)
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


def get_text_matching(ref_text, eva_text, min_size=1):
    """
    进行文本匹配
    :param ref_text:
    :param eva_text:
    :param min_size:
    :return:
    """
    matches = difflib.SequenceMatcher(None, eva_text, ref_text).get_matching_blocks()
    result = []
    for match in matches:
        if match.size >= min_size:
            output = eva_text[match.a:match.a + match.size]
            result.append({"text": output, "index": match.a, "offset": match.size})
    return result


def get_keyword_info(text: str, keyword: str, synonym: list):
    """
    :param text:
    :param keyword:
    :param synonym:
    :return:
    """
    info = {keyword: text.count(keyword)}
    for group in synonym:
        # 判断keyword是否是同义词中
        if keyword in group:
            info.update({w: text.count(w) for w in group})
            break
    return info


def get_text_similarity(str1, str2):
    """
    计算文本相似度
    :param str1:
    :param str2:
    :return: simi 相似度
             dist 编辑距离
    """
    dist = editdistance.eval(str1, str2)
    size = max(len(str1), len(str2))
    simi = 0.0
    if size > 0: simi = 1. - dist / size
    return simi, dist


def get_text_similarity_keyword(ref_text, eva_text, keyword={}, synonym=[]):
    """
    计算文本相似度
    :param ref_text: 参考文本，标准文本
    :param eva_text: 测评文本，待匹配文本
    :param keyword:  关键词权重{"text1": weight1,"text2": weight2,...}
    :param synonym:  同义词列表,如[["开心", "高兴", "愉快"], ["今天", "当天"]]
    :return:
    """
    size = max(len(ref_text), len(eva_text))
    simi, dist = get_text_similarity(ref_text, eva_text)
    m = simi / size if size > 0 else 0
    keys = {}
    for k, w in keyword.items():
        key_info = get_keyword_info(text=eva_text, keyword=k, synonym=synonym)
        num = sum([len(k) * n for k, n in key_info.items()]) if key_info else 0
        score = w if num > 0 else 0.0
        keys[k] = {"nums": num, "score": score}
    keys_nums = [s['nums'] for s in list(keys.values())]
    keys_score = [s['score'] for s in list(keys.values())]
    other_nums = size - sum(keys_nums)
    # other_w = 1.0 - sum(keyword.values()) # BUG
    other_w = 1.0 - sum(keys_score)
    other_score = other_w * m * other_nums
    total_score = other_score + sum(keys_score)
    result = {"ref_text": ref_text, "eva_text": eva_text, "similarity": simi,
              "score": total_score, "other_score": other_score}
    return result


def get_text_matching_similarity(ref_text, eva_text, keyword={}, synonym=[], ignore_words=[]):
    """
    :param ref_text: 参考文本，标准文本
    :param eva_text: 测评文本，待匹配文本
    :param keyword:  关键词权重{"text1": weight1,"text2": weight2,...}
    :param synonym:  同义词列表,如[["开心", "高兴", "愉快"], ["今天", "当天"]]
    :param ignore_words:  需要忽略的字词
    :return:
    """
    match_info = get_text_matching(ref_text, eva_text)
    # 去除非中文字符
    ref_text = get_chinese_text(ref_text, repl="")
    eva_text = get_chinese_text(eva_text, repl="")
    # 去除忽略字词
    if ignore_words:
        ref_text = del_ignore_words(ref_text, ignore_words=ignore_words)
        eva_text = del_ignore_words(eva_text, ignore_words=ignore_words)
    result = get_text_similarity_keyword(ref_text, eva_text, keyword=keyword, synonym=synonym)
    # 同义词文本
    synonym_texts = get_synonym_texts(text=eva_text, synonym=synonym)
    synonym_result = []
    for text in synonym_texts:
        res = get_text_similarity_keyword(ref_text, text, keyword=keyword, synonym=synonym)
        synonym_result.append(res)
    result['similarity'] = max([r['similarity'] for r in synonym_result])
    result['other_score'] = max([r['other_score'] for r in synonym_result])
    result['score'] = max([r['score'] for r in synonym_result] + [result['similarity']])
    result['match_info'] = match_info
    return result


def get_standard_text(text, ignore_words):
    """
    获得标准文本，用于测评
    :param text:
    :return:
    """
    text = del_ignore_words(text, ignore_words=ignore_words, repl=" ")  # 去除忽略词
    text = del_punctuation(text, repl=" ")  # 去除标点符号
    text = rep_text(text, words=[" "], repl="，", unique=True)  # 将标点符号统一转为，
    return text


if __name__ == "__main__":
    ignore_words = ["呃", "啊", "嗯", "哦"]
    string1 = 'OK,我是一名程序员，穿工作服'
    string1 = 'OK,我是一名程序员，穿工作服'
    string1 = '程序员'
    # string2 = '我是一名IT程序员，喜欢穿工作服' # (simi,score)= (1.0, 0.86875)
    # string2 = '我是一名IT程序猿，喜欢穿格子衫'  # (simi,score)= (0.9375, 0.83320312)
    string1 = '报告考评员，本次工作任务已完成，人员已撤离。'
    string2 = "啊啊啊啊啊，报告考评员，呃，左右，呃，作业已完成，呃，人员已撤"

    # keyword = {"程序猿": 0.5, "工作服": 0.3, "安全帽": 0.1}  # (simi,score)= (0.9375, 0.9171875)
    keyword = {"程序员": 0.5, "工作服": 0.3, "安全帽": 0.1}  # (simi,score)= (0.9375, 0.833203125)
    # keyword = {"工作服": 0.3, "安全帽": 0.1}  # (simi,score)= (0.9375, 0.833203125)
    # keyword = {}
    # synonym = []
    synonym = [["程序猿", "程序员"], ["衣服", "格子衫", "工作服"]]
    # string1 = '123456789'
    # string2 = 'ABCD'
    # print("---" * 10)
    # print(get_text_matching(string1, string2))
    # print("---" * 10)
    # # print(get_text_similarity_keyword(string1, string2, keyword={}))
    # print("---" * 10)
    # print(get_text_similarity_keyword(string1, string2, keyword=keyword, synonym=synonym))
    # print("---" * 10)
    # print(get_text_matching_similarity(string1, string2, keyword=keyword, synonym=synonym, ignore_words=ignore_words))
    text = "，  你你好好，，，呀,,,  。"
    text = get_standard_text(text, ignore_words=ignore_words)
    print(text)
