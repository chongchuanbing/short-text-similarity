# -*- coding:utf-8 -*-

import jieba

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./Params/stop_words_ch.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence, remove_stopwords=False):
    # 对文档中的每一行进行中文分词

    if float == type(sentence):
        return ''

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

def character_depart(sentence, remove_stopwords=False):

    character_arr = None

    if remove_stopwords:
        # 创建一个停用词列表
        stopwords = stopwordslist()

        character_arr = [char for char in sentence if char not in stopwords]
    else:
        character_arr = [char for char in sentence]

    return ' '.join(character_arr)