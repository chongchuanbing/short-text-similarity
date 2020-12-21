# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import csv
import codecs
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.preprocessing.text

import _pickle as cPickle

from tools.config_util import ConfigParser
from tools import jieba_util
from core.loss_history import LossHistory

from core.model.BiLstm_siamese_model import BiLstm_siamese_model
from core.model.Simple_lstm_cnn_model import Simple_lstm_cnn_model
from core.model.Multi_lstm_cnn_model import Multi_lstm_cnn_model

from core.datadeal.Word_data_deal import Word_data_deal
from core.datadeal.Character_data_deal import Character_data_deal

import sys
from imp import reload
reload(sys)


class Lstm(object):
    def __init__(self):
        ########################################
        # set directories and parameters
        ########################################
        self.DATA_DIR = './data/'
        self.MAX_SEQUENCE_LENGTH = 30
        self.MAX_NB_WORDS = 200000
        self.EMBEDDING_DIM = 300

        self.num_lstm = 175
        self.num_dense = 100
        self.rate_drop_lstm = 0.15
        self.rate_drop_dense = 0.15

        self.STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.num_lstm, self.num_dense, self.rate_drop_lstm, \
                                                        self.rate_drop_dense)
        self.save_path = "./Params"

        con = ConfigParser()
        self.__lstm_vector_key = con.get_config('lstm', 'lstm_vector_key')
        self.__lstm_model_key = con.get_config('lstm', 'lstm_model_key')

        if 'word' == self.__lstm_vector_key:
            # self.EMBEDDING_FILE = './Params/baike_26g_news_13g_novel_229g.bin'
            self.EMBEDDING_FILE = './Params/word_w2v.mod'
            self.tokenizer_name = con.get_config('tokenizer', 'word_tokenizer_name')
            self.word_dict_name = 'word_tokenizer.npy'
            self.embedding_matrix_path = "./Params/word_embedding_matrix.npy"
            self.vocab_path = './Params/word_vocabs.txt'

            self.data_deal = Word_data_deal()
            pass
        elif 'character' == self.__lstm_vector_key:
            self.EMBEDDING_FILE = './Params/character_w2v.mod'
            self.tokenizer_name = con.get_config('tokenizer', 'character_tokenizer_name')
            self.word_dict_name = 'character_tokenizer.npy'
            self.embedding_matrix_path = "./Params/character_embedding_matrix.npy"
            self.vocab_path = './Params/character_vocabs.txt'

            self.data_deal = Character_data_deal()
            pass

        self.toeknizer_path = os.path.join(self.save_path, self.tokenizer_name)
        self.word_dict_path = os.path.join(self.save_path, self.word_dict_name)

        # 创建一个实例history
        self.__loss_history = LossHistory()

        pass

    def get_model(self,
                  nb_words,
                  embedding_matrix):
        if 'simple_lstm_cnn_model' == self.__lstm_model_key:
            simple_lstm_cnn_model = Simple_lstm_cnn_model(vocab_size=nb_words,
                                                        embedding_dim=self.EMBEDDING_DIM,
                                                        sequence_length=self.MAX_SEQUENCE_LENGTH,
                                                        embedding_matrix=embedding_matrix)
            model = simple_lstm_cnn_model.model_create()
        elif 'bilstm_siamese_model' == self.__lstm_model_key:
            bilstm_siamese_model = BiLstm_siamese_model(vocab_size=nb_words,
                                                        embedding_dim=self.EMBEDDING_DIM,
                                                        sequence_length=self.MAX_SEQUENCE_LENGTH,
                                                        embedding_matrix=embedding_matrix)
            model = bilstm_siamese_model.model_create()
        elif 'multi_lstm_cnn' == self.__lstm_model_key:
            multi_lstm_cnn_model = Multi_lstm_cnn_model(vocab_size=nb_words,
                                                        embedding_dim=self.EMBEDDING_DIM,
                                                        sequence_length=self.MAX_SEQUENCE_LENGTH,
                                                        embedding_matrix=embedding_matrix)
            model = multi_lstm_cnn_model.model_create()

        return model

    def get_model_name(self):
        return self.save_path + '/' + self.__lstm_model_key  + '_' + self.STAMP

    def __text_to_word_sequence(self, text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
        if lower:
            text = text.lower()

        # if type(text) == unicode:
        #     translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
        # else:
        #     translate_table = keras.maketrans(filters, split * len(filters))
        translate_table = text.maketrans(filters, split * len(filters))
        text = text.translate(translate_table)
        seq = text.split(split)
        return [i for i in seq if i]

    def __load_tokenizer(self):
        return cPickle.load(open(os.path.join(self.save_path, self.tokenizer_name), 'rb'))

    def train(self,
              file_path_arr,
              enhancement_file_path_arr,
              epochs,
              batch_size,
              validation_split,
              enhancement_split):

        df_train, df_val = self.data_deal.load_data(file_path_arr, validation_split)

        if len(enhancement_file_path_arr) > 0:
            df_train = self.data_deal.load_enhancement(df_train, enhancement_file_path_arr, enhancement_split)

        ########################################
        # prepare embeddings
        ########################################
        print('Preparing embedding matrix')
        print(self.EMBEDDING_FILE)

        vocabs_train = self.data_deal.build_vocab(df_train)
        vocabs_val = self.data_deal.build_vocab(df_val)

        vocabs_all = vocabs_train.union(vocabs_val)

        word_index, word_vector, embeddings_matrix = self.data_deal.word2vec_dictionaries2(vocabs_all, self.EMBEDDING_FILE, self.EMBEDDING_DIM)

        data_1_train, data_2_train, labels_train = self.data_deal.data_transform2(word_index, df_train, self.MAX_SEQUENCE_LENGTH)
        data_1_val, data_2_val, labels_val = self.data_deal.data_transform2(word_index, df_val, self.MAX_SEQUENCE_LENGTH)

        labels_train = np.expand_dims(labels_train, 2)
        labels_val = np.expand_dims(labels_val, 2)

        print(data_1_train.shape)
        print(data_2_train.shape)

        print('Found %s train texts, val texts: %s' % (len(data_1_train), len(data_1_val)))

        nb_words = min(self.MAX_NB_WORDS, len(word_index)) + 1

        cPickle.dump(word_index, open(self.word_dict_path, "wb"))
        np.save(self.embedding_matrix_path, embeddings_matrix)

        model_name = self.get_model_name()
        print(model_name)

        model = self.get_model(nb_words, embeddings_matrix)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        bst_model_path = model_name + '.h5'

        # 'model - ep{epoch: 03d} - loss{loss: .3f} - val_loss{val_loss: .3f}.h5'

        model_checkpoint = ModelCheckpoint(
            bst_model_path,
            # monitor='val_acc',
            monitor='val_loss',
            mode='min',
            period=1,
            save_best_only=True,
            save_weights_only=True)

        tensorboard = TensorBoard(log_dir='logs')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

        # hist = model.fit([data_1, data_2], labels, \
        #                  validation_data=([data_1, data_2], labels), \
        #                  epochs=100, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])

        hist = model.fit(
            x=[data_1_train, data_2_train],
            y=labels_train,
            validation_data=([data_1_val, data_2_val], labels_val),

            # x=[data_1, data_2],
            # y=labels,
            # validation_split=validation_split,

            epochs=epochs,
            batch_size=batch_size,
            # shuffle=True,
            # callbacks=[early_stopping, model_checkpoint],
            # callbacks=[model_checkpoint, tensorboard, reduce_lr, self.__loss_history]
        )

        # self.__loss_history.loss_plot('epoch')

        model.save_weights(bst_model_path)

        # bst_score = min(hist.history['loss'])
        # bst_acc = max(hist.history['acc'])
        # print(bst_acc, bst_score)

        pass

    def validate(self, file_path,
                 batch_size):

        texts_1 = []
        texts_2 = []
        count = 0
        with codecs.open(file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for values in reader:
                texts_1.append(self.data_deal.parse_sentence(values[0]))
                texts_2.append(self.data_deal.parse_sentence(values[1]))
                count += 1
        print('Found %s texts in train.csv' % len(texts_1))

        print('Preparing embedding matrix')

        word_index = cPickle.load(open(open(self.word_dict_path, "rb")))
        embeddings_matrix = np.load(self.embedding_matrix_path)

        data_1 = self.data_deal.word2vec_tokenizer(word_index, texts_1)
        data_2 = self.data_deal.word2vec_tokenizer(word_index, texts_2)

        nb_words = min(self.MAX_NB_WORDS, len(word_index)) + 1

        model = self.get_model(nb_words, embeddings_matrix)

        model_name = self.get_model_name()
        print(model_name)

        bst_model_path = model_name + '.h5'

        model.load_weights(bst_model_path)

        predicts = model.predict([data_1, data_2], batch_size=batch_size, verbose=1)
        for i in range(count):
            print("t1: %s, t2: %s, score: %.8f" % (''.join(texts_1[i]), ''.join(texts_2[i]), predicts[i][0]))

    def validate_1_N(self, base_storehouse_path,
                     batch_size):
        texts_2 = []
        count = 0
        with codecs.open(base_storehouse_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for values in reader:
                texts_2.append(self.data_deal.parse_sentence(values[0]))
                count += 1

        print('base storehouse count: ' + str(count))

        print('Preparing embedding matrix')
        # word2vec = Word2Vec.load(self.EMBEDDING_FILE)
        # # word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE, binary=True, unicode_errors='ignore')
        # word_index, word_vector, embeddings_matrix = self.__word2vec_dictionaries(word2vec)

        # word_index = np.load(self.word_dict_path)
        word_index = cPickle.load(open(self.word_dict_path, "rb"))
        embeddings_matrix = np.load(self.embedding_matrix_path)
        # word_index, word_vector, embeddings_matrix = self.__word2vec_dictionaries2()

        data_2 = self.data_deal.word2vec_tokenizer(word_index, texts_2, self.MAX_SEQUENCE_LENGTH)

        nb_words = min(self.MAX_NB_WORDS, len(word_index)) + 1

        model = self.get_model(nb_words, embeddings_matrix)

        model_name = self.get_model_name()
        print(model_name)

        bst_model_path = model_name + '.h5'

        model.load_weights(bst_model_path)

        while True:
            text = input('输入短句: ')
            text_1 = [self.data_deal.parse_sentence(text)]

            for index in range(count - 1):
                text_1.append(text_1[0])

            data_1 = self.data_deal.word2vec_tokenizer(word_index, text_1, self.MAX_SEQUENCE_LENGTH)

            print(len(data_1))
            print(len(data_2))

            time1 = time.time()

            result_arr = []

            predicts = model.predict([data_1, data_2], batch_size=batch_size, verbose=1)
            for i in range(count):
                # print("t1: %s, t2: %s, score: %.8f" % (''.join(text_1[i]), ''.join(texts_2[i]), predicts[i][0]))

                result_arr.append({
                    't1': (''.join(text_1[i])),
                    't2': (''.join(texts_2[i])),
                    'score': '%.8f' % (predicts[i][0])
                })

            result_arr = sorted(result_arr, key=lambda x: x['score'], reverse=True)

            for x in result_arr[: 10]:
                print(x)

            print('costTime: %4f' % (time.time() - time1))

    def print_w2v(self, file_path):

        texts_1 = []
        texts_2 = []
        count = 0
        with codecs.open(file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for values in reader:
                texts_1.append(self.data_deal.parse_sentence(values[0]))
                texts_2.append(self.data_deal.parse_sentence(values[1]))
                count += 1

        keras.preprocessing.text.text_to_word_sequence = self.__text_to_word_sequence
        '''
        end here
        '''

        print('Load tokenizer...')
        tokenizer = self.__load_tokenizer()

        sequences_1 = tokenizer.texts_to_sequences(texts_1)
        sequences_2 = tokenizer.texts_to_sequences(texts_2)

        for index in range(len(texts_1)):
            print('q1: %s, %s' % (texts_1[index], sequences_1[index]))
            print('q2: %s, %s' % (texts_2[index], sequences_2[index]))
