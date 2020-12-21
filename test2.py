# /usr/bin/env python
# coding=utf8

import http.client
import hashlib
from urllib import request
import random
import json
import csv
import time
import pandas as pd

appid = ''  # 你的appid
secretKey = ''  # 你的密钥

def baidu_translate(sentence):
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = sentence
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf8"))
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + request.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    result = response.read()
    result_json = json.loads(result)

    print(result_json)

    translate_str = result_json['trans_result'][0]['dst']

    return translate_str


# with open('./data/msr_label.csv', "r") as f, open('./data/msr_label_translate.csv', 'a+') as f2:
#
#     f_csv = csv.reader(f)
#     next(f_csv, None)
#
#     w_csv = csv.writer(f2)
#
#     for row in f_csv:
#         similay = row[0]
#         sentence1 = row[1]
#         sentence2 = row[2]
#
#         print(similay, sentence1, sentence2)
#
#         sentence1_translate = baidu_translate(sentence1)
#
#         time.sleep(1)
#
#         sentence2_translate = baidu_translate(sentence2)
#
#         w_csv.writerow([sentence1_translate, sentence2_translate, similay])
#
#         time.sleep(1)


train = pd.read_table('datadeal/knowledge.csv', header=None, encoding='utf-8-sig', sep=',')

train.reset_index(inplace=True)

train[0].to_frame().to_csv('datadeal/knowledge_question.csv', header=0, index=0)



# if httpClient:
#     httpClient.close()