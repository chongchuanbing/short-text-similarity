# -*- coding:utf-8 -*-

import codecs
import os
import jieba

from keras.preprocessing.text import Tokenizer

import _pickle as cPickle



DATA_DIR = './data/'
TRAIN_DATA_FILE = DATA_DIR + 'words.txt'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000

save_path = "./Params"
tokenizer_name = "tokenizer.pkl"


# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./Params/stop_words_ch.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence, remove_stopwords=False):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip(), cut_all=False)

    if remove_stopwords:
        # 创建一个停用词列表
        stopwords = stopwordslist()
        # 输出结果为outstr

        word_arr = []
        for word in sentence_depart:
            if word not in stopwords:
                if word != '\t':
                    word_arr.append(word)
        return ' '.join(word_arr)

    return ' '.join(sentence_depart)

words = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    line_content = f.readline()
    while line_content:
        words.append(seg_depart(line_content, True))

        line_content = f.readline()

print('Found %s texts in train.csv' % len(words))

print("Fit tokenizer...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
tokenizer.fit_on_texts(words)
print("Save tokenizer...")
if not os.path.exists(save_path):
    os.makedirs(save_path)
cPickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))

