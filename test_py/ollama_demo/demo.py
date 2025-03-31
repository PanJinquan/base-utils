# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-03-25 11:09:27
# @Brief  : https://www.runoob.com/ollama/ollama-commands.html
# --------------------------------------------------------
"""
from ollama import Client

host = "http://192.168.68.102:50000"
client = Client(host=host, headers={'x-some-header': 'some-value'})


def chat_example(input="你说谁？"):
    r = client.chat(model='deepseek-r1:7b', messages=[{'role': 'user', 'content': input, }, ])
    print(r['message']['content'])


def embed_example(input=["你说", "这是什么"]):
    r = client.embed(model='nomic-embed-text:latest', input=input)  # 新版本
    # r = client.embeddings(model='nomic-embed-text:latest', prompt=input[0]) # 旧版本不支持输入多个
    for i in range(len(input)):
        print(input[i], r['embeddings'][i])


if __name__ == '__main__':
    # chat_example()
    embed_example()
