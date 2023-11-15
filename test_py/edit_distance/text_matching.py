# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-11-09 11:20:04
    @Brief  :
"""
import re, string
from typing import Dict, List
from test_py.edit_distance import text_utils


class TextMatching(object):
    def __init__(self):
        self.ignore_words = ["呃", "啊", "嗯", "哦"]

    def get_text_similarity(self, data):
        """
        进行文本相似度比较
        :param data:
        :return:
        """
        ref_text = data.get('text', "")
        asr_text = data.get('asr_text', "")
        result = {'text': ref_text,
                  'asr_text': asr_text,
                  'asr_text_window': asr_text,
                  'match_info': [],
                  'similarity': 0.0,
                  'other_score': 0.0,
                  'score': 0.,
                  'code': 0,
                  'msg': "OK"
                  }
        try:
            threshold = data.get('similarity_threshold', 0.6)
            # 获得标准文本，用于测评
            ref_text = text_utils.get_standard_text(ref_text, ignore_words=self.ignore_words)
            eva_text = text_utils.get_standard_text(asr_text, ignore_words=self.ignore_words)
            config = data.get('config', {})
            keyword = {word['keyword']: word['weight'] for word in config.get('keywords', [])}
            synonym = config.get('synonym', [])
            res = text_utils.get_text_matching_similarity(ref_text=ref_text,
                                                          eva_text=eva_text,
                                                          keyword=keyword,
                                                          synonym=synonym)
            match_info = []
            for info in res["match_info"]:
                r = {'asr_window_start': info['index'],
                     'asr_window_offset': info['offset'],
                     'match': info['text'],
                     'weight': None,
                     'score': None}
                match_info.append(r)

            score = res['score'] if res['score'] > threshold or res['similarity'] > threshold else 0.0
            result['asr_text_window'] = eva_text
            result['similarity'] = res['similarity']
            result['other_score'] = res['other_score']
            result['score'] = score
            result['match_info'] = match_info
        except Exception as e:
            result.update({'similarity': 0.0,
                           'other_score': 0.0,
                           'score': 0.,
                           'match_info': [],
                           'code': 5001,
                           'msg': "Error: {}".format(e)})
        return result

    def task_infer(self, payload: List[Dict]):
        results = []
        for data in payload:
            res = self.get_text_similarity(data)
            results.append(res)
        return results


if __name__ == "__main__":
    from pybaseutils import json_utils, image_utils, base64_utils

    m = TextMatching()

    data = \
        {
            "data": [
                {
                    "text": "报告考评员，本次工作任务已完成，人员已撤离。",
                    "asr_text": "啊啊啊啊啊，报告考评员，呃，左右，呃，作业已完成，呃，人员已撤",
                    "config": {}
                },
                {
                    "text": "工作服穿着整齐，扣子扣好。",
                    "asr_text": "工作服比较整齐，纽扣扣好。",
                    "config": {
                        "keywords": [
                            {
                                "keyword": "工作服",
                                "weight": 0.6
                            },
                            {
                                "keyword": "整齐",
                                "weight": 0.1
                            },
                            {
                                "keyword": "扣子",
                                "weight": 0.1
                            },
                            {
                                "keyword": "纽扣",
                                "weight": 0.1
                            }
                        ],
                        "synonym": [["扣子", "纽扣"], ["整齐", "整洁", "干净"]]
                    },
                }
            ],
            "reqid": "0d5c31abe59a49aea31f264fd58bcf40"
        }

    data = {
        "data": [
            {
                # "text": "OK,我是一名程序猿，穿工作服",
                # "text": "我是一名IT程序员，喜欢穿工作服",
                # "asr_text": "我是一名IT程序猿，喜欢穿格子衫",
                "text": "",
                "asr_text": " ",
                "config": {
                    "keywords": [
                        {
                            "keyword": "程序猿",
                            "weight": 0.5
                        },
                        {
                            "keyword": "工作服",
                            "weight": 0.3
                        },
                        {
                            "keyword": "安全帽",
                            "weight": 0.1
                        },
                    ],
                    "synonym": [["程序员", "程序猿"], ["工作服", "格子衫"]]
                }
            },
        ],
        "reqid": "0d5c31abe59a49aea31f264fd58bcf40"
    }

    result = m.task_infer(payload=data["data"])
    print(json_utils.formatting(result))
