# -*- coding:utf-8 -*-

import logging
import multiprocessing
from gensim.models.word2vec import LineSentence, Word2Vec
from tools.config_util import ConfigParser


class CharacterVector(object):

    def __init__(self,
                 min_count=5,
                 window=5,
                 iter=50,
                 epochs=50):
        '''
        最小字数
        迭代轮数
        :param min_count:
        :param iter:
        :param epochs:
        '''
        self.__min_count = min_count
        self.__iter = iter
        self.__epochs = epochs
        self.__window = window

        self.__train_cores = multiprocessing.cpu_count()

        con = ConfigParser()
        self.__params_dir = con.get_config('common', 'params_dir')
        self.__w2v_name = con.get_config('w2v', 'character_w2v_name')
        self.__model_path = "./%s/%s" % (self.__params_dir, self.__w2v_name)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        pass

    def __load_data(self):
        self.__characters = LineSentence("./data/character.txt")

    def train(self):
        self.__load_data()

        model = Word2Vec(self.__characters, min_count=self.__min_count, size=300, iter=self.__iter, workers=self.__train_cores, window=self.__window)
        model.wv.save_word2vec_format(self.__model_path, binary=False)

        # model.train(self.__characters, total_examples=model.corpus_count, epochs=self.__epochs)
        # model.save(self.__model_path)
        pass

    def validate(self):
        model_loaded = Word2Vec.load(self.__model_path)
        sim = model_loaded.wv.most_similar(positive=[u'艾佳贷'])
        for s in sim:
            print(s[0])



