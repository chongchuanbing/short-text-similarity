# -*- coding:utf-8 -*-

import csv
import re

pattern = re.compile(r'<[^>]+>',re.S)

def html_text(sentence):
    sentence = pattern.sub('', sentence)

    sentence = sentence.replace('\t', '')
    sentence = sentence.replace('\r', '')
    sentence = sentence.replace('\n', '')

    return sentence

# with open('./data/db_export.txt', "r") as f, open('./data/sentence.csv', 'a+') as f2:
#     line_content = f.readline()
#
#     while line_content:
#
#         print(line_content)
#
#         line_content = line_content.replace('<script[^>]*?>[\s\S]*?<\/script>', '')
#         line_content = line_content.replace('<style[^>]*?>[\s\S]*?<\/style>', '')
#         line_content = line_content.replace('<[^>]+>', '')
#         line_content = line_content.replace('\\s*|\t|\r|\n', '')
#
#         words = jieba_util.seg_depart(line_content)
#         if len(words) > 0:
#             f2.write(words + '\n')
#
#         line_content = f.readline()

with open('./data/knowledge.csv', "r") as f, open('./data/sentence.csv', 'w+') as f2:
    f_csv = csv.reader(f)

    f2_csv = csv.writer(f2)
    for row in f_csv:
        for row_item in row:

            # print(row_item)

            row_item = html_text(row_item)

            if '' == row_item:
                continue

            print(row_item)

            f2_csv.writerow([row_item])

with open('./data/train.csv', "r") as f, open('./data/sentence.csv', 'a+') as f2:
    f_csv = csv.reader(f)

    f2_csv = csv.writer(f2)
    for row in f_csv:
        sentence1 = row[0]
        sentence2 = row[1]

        sentence1 = html_text(sentence1)

        if '' != sentence1:
            print(sentence1)
            f2_csv.writerow([sentence1])

        sentence2 = html_text(sentence2)

        if '' != sentence2:
            print(sentence2)
            f2_csv.writerow([sentence2])

with open('./data/train_atec.csv', "r") as f, open('./data/sentence.csv', 'a+') as f2:
    f_csv = csv.reader(f)

    f2_csv = csv.writer(f2)
    for row in f_csv:
        sentence1 = row[0]
        sentence2 = row[1]

        sentence1 = html_text(sentence1)

        if '' != sentence1:
            print(sentence1)
            f2_csv.writerow([sentence1])

        sentence2 = html_text(sentence2)

        if '' != sentence2:
            print(sentence2)
            f2_csv.writerow([sentence2])


