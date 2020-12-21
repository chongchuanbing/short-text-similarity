# -*- coding:utf-8 -*-

import csv
import jieba
import re

from tools import jieba_util

with open('./data/key_words.txt', "r") as f:
    key_word = f.readline()
    while key_word:
        jieba.suggest_freq(key_word, True)

        key_word = f.readline()

pattern = re.compile(r'<[^>]+>',re.S)

def row_write_csv(f, sentence):
    sentence = pattern.sub('', sentence)

    sentence = sentence.replace('\t', '')
    sentence = sentence.replace('\r', '')
    sentence = sentence.replace('\n', '')

    words = jieba_util.seg_depart(sentence)

    print(words)

    if len(words) > 0:
        f.write(words + '\n')

with open('./data/sentence.csv', "r") as f, open('./data/words.txt', 'a+') as f2:
    f_csv = csv.reader(f)

    for row in f_csv:
        sentence1 = row[0]

        row_write_csv(f2, sentence1)