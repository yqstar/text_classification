import pandas as pd
import jieba
import numpy as np
from collections import Counter


class CorpusDict(object):
    def __init__(self, corpus_path, stopwords_path, sp_words=["unk", "pad", "num"]):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.corpus = pd.read_csv(corpus_path)
        self.word2freq = Counter(sp_words)
        self.idx2word = list(self.word2freq.keys())
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.length = len(sp_words)

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        self.word2freq.update([word])
        self.idx2word = list(dict(self.word2freq).keys())
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.length
            self.length += 1
        return self.word2idx[word]

    def text_vector(self, text):
        return [self.word2idx[word] if word in self.word2idx else self.word2idx["unk"] for word in self.token(text)]

    def onehot_encoded(self, word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]] = 1
        return list(vec)

    def random_embedding(self, word, embedding_dim=300):
        matrix_shape = (self.length, embedding_dim)
        embedding_lookup_matrix = np.random.random(matrix_shape)
        return list(embedding_lookup_matrix[self.word2idx[word], :])

    def write_dict(self):
        with open(self.corpus_path, 'w', encoding='utf-8') as f:
            f.write(str(self.word2idx))
        print("Corpus Dict is saved successfully!")

    def read_dict(self, dict_path=None):
        if not dict_path:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                # eval函数将字符串形式的字典转成字典
                dict_corpus = eval(f.read())
        else:
            with open(dict_path, 'r', encoding='utf-8') as f:
                # eval函数将字符串形式的字典转成字典
                dict_corpus = eval(f.read())
        return dict_corpus

    @staticmethod
    def token(text, token_level='word'):
        if token_level == "word":
            tokens = jieba.lcut(text)
        elif token_level == "char":
            tokens = list(text)
        return tokens

    # N-gram表示,可用于拼写校正和文本摘要
    @staticmethod
    def n_gram(tokens, n=2):
        return [tuple(tokens[idx:idx + n]) for idx in range(len(tokens) - n + 1)]

    def remove_stopwords(self):
        pass



    # @staticmethod
    # def build_dict(self, corpus, min_frequency=5):
    #     # 语料库构建词频
    #     word_frequency = dict(Counter(corpus))
    #     word2idx = {word: idx for idx, word in enumerate(word_frequency.keys())}
    #     word2idx['unk'] = len(word2idx)
    #     word2idx['pad'] = len(word2idx)
    #     idx2word = {idx: word for word, idx in word2idx.items()}
    #     # 字典保存到本地
    #     with open("dict.txt", 'w', encoding='utf-8') as f:
    #         f.write(str(word2idx))
    #     return word2idx, idx2word

    # @staticmethod
    # def text_vector(text, word2idx, max_length):
    #     return [word2idx[i] for i in list(text)][0:max_length]


# if __name__ == '__main__':
#     text = pd.read_csv("data/all_data.tsv", sep="\t")
#     corpus = ''
#     for item in text.text:
#         corpus += item
#     word2idx, idx2word = build_dict(corpus)

