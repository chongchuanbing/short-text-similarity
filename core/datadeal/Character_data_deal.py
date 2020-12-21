# -*- coding:utf-8 -*-


from core.datadeal.Base_data_deal import Base_data_deal

class Character_data_deal(Base_data_deal):

    def __init__(self):
        pass

    def parse_sentence(self, sentence):
        return [char for char in sentence if char]