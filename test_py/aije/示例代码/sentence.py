# -*- coding: utf-8 -*-
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class SentenceBERT:
    def __init__(self, model_file):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_file, device=self.device)
        self.model.eval()

    def encode(self, sentences, normalize_embeddings: bool = True):
        return self.model.encode(sentences,
                                 convert_to_tensor=False,
                                 normalize_embeddings=normalize_embeddings)

    def similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的语义相似度"""
        emb1 = self.encode(sent1)
        emb2 = self.encode(sent2)
        return np.dot(emb1, emb2)

if __name__ == "__main__":
    model_file = 'paraphrase-multilingual-MiniLM-L12-v2'
    infer = SentenceBERT(model_file)
    score = infer.similarity("检查安全帽", "我检查了安全帽")
    print("相似度:", score)
