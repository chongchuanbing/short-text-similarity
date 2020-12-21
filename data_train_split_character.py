# -*- coding:utf-8 -*-

import csv

from tools import jieba_util

with open('./data/sentence.csv', "r") as f, open('./data/character.txt', 'a+') as f2:
    f_csv = csv.reader(f)

    for row in f_csv:
        sentence1 = row[0]

        sentence1_word_arr = jieba_util.character_depart(sentence1)

        if len(sentence1_word_arr) > 0:
            f2.write(' '.join(sentence1_word_arr) + '\n')