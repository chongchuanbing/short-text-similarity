# -*- coding:utf-8 -*-


from core.datadeal.Base_data_deal import Base_data_deal

from tools import jieba_util

class Word_data_deal(Base_data_deal):

    def __init__(self):
        pass

    def parse_sentence(self, sentence):
        return jieba_util.seg_depart(sentence)
