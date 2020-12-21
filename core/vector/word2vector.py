# -*- coding:utf-8 -*-

import logging
import multiprocessing
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from tools.config_util import ConfigParser


class WordVector(object):

    def __init__(self,
                 min_count=1,
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

        self.__train_cores = multiprocessing.cpu_count()

        con = ConfigParser()
        self.__params_dir = con.get_config('common', 'params_dir')
        self.__w2v_name = con.get_config('w2v', 'word_w2v_name')
        self.__model_path = "./%s/%s" % (self.__params_dir, self.__w2v_name)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        pass

    def __load_data(self):
        self.__sentences = LineSentence("./data/words.txt")

    def __create_dictionaries(self,
                            model=None,
                            combined=None,
                            maxlen=100):
        ''' Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
        '''
        if (combined is not None) and (model is not None):
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(),
                                allow_update=True)
            # the index of a word which have word vector is not 0
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}
            # integrate all the corresponding word vectors into the word vector matrix
            w2vec = {word: model[word] for word in w2indx.keys()}

            # a word without a word vector is indexed 0,return the index of word
            def parse_dataset(combined):
                ''' Words become integers
                '''
                data = []
                for sentence in combined:
                    new_txt = []
                    for word in sentence:
                        try:
                            new_txt.append(w2indx[word])
                        except:
                            new_txt.append(0)
                    data.append(new_txt)
                return data

            combined = parse_dataset(combined)
            # unify the length of the sentence with the pad_sequences function of keras
            combined = sequence.pad_sequences(combined, maxlen=maxlen)
            # return index, word vector matrix and the sentence with an unifying length and indexed
            return w2indx, w2vec, combined
        else:
            print('No datadeal provided...')

    def train(self):
        self.__load_data()

        model = Word2Vec(self.__sentences, min_count=self.__min_count, iter=self.__iter, workers=self.__train_cores)
        model.train(self.__sentences, total_examples=model.corpus_count, epochs=self.__epochs)

        model.save(self.__model_path)

        # index, word vector matrix and the sentence with an unifying length and indexed based on the trained model
        # index_dict, word_vectors, combined = self.__create_dictionaries(model=model, combined=combined)
        #
        # return index_dict, word_vectors, combined

    def validate(self):
        model_loaded = Word2Vec.load(self.__model_path)
        sim = model_loaded.wv.most_similar(positive=[u'艾佳贷'])
        for s in sim:
            print(s[0])

