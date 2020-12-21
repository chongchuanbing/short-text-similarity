# -*- coding:utf-8 -*-

import configparser

class ConfigParser():

    config_dic = {}

    @classmethod
    def get_config(cls, sector, item):
        value = None
        try:
            value = cls.config_dic[sector][item]
        except KeyError:
            cf = configparser.ConfigParser()

            # 注意setting.ini配置文件的路径
            cf.read('config.ini', encoding='utf8')
            value = cf.get(sector, item)
            cls.config_dic[sector][item] = value
        finally:
            return value