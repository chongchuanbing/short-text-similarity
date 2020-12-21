# -*- coding:utf-8 -*-


from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Lambda, CuDNNLSTM, Conv1D, MaxPool1D, Concatenate, MaxPooling1D, GlobalMaxPool1D, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.preprocessing.text
from keras import backend as K

import tensorflow as tf

from core.model.Base_model import Base_model


class BiLstm_siamese_model(Base_model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 sequence_length,
                 embedding_matrix
                 ):
        self.__VOCAB_SIZE = vocab_size;
        self.__EMBEDDING_DIM = embedding_dim;
        self.__SEQUENCE_LENGTH = sequence_length;
        self.__EMBEDDING_MATRIX = embedding_matrix;
        pass

    def __exponent_neg_manhattan_distance2(self, inputX):
        (sent_left, sent_right) = inputX
        '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    def __exponent_neg_manhattan_distance(self, sent_left, sent_right):
        '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    '''基于欧式距离的字符串相似度计算'''
    def __euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def __create_base_network(self, input_shape):
        '''搭建编码层网络,用于权重共享'''

        gpu_enable = tf.test.is_gpu_available(
            cuda_only=False,
            min_cuda_compute_capability=None
        )

        if gpu_enable:
            input = Input(shape=input_shape)
            lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input)
            lstm1 = Dropout(0.5)(lstm1)
            lstm2 = Bidirectional(CuDNNLSTM(32))(lstm1)
            lstm2 = Dropout(0.5)(lstm2)
        else:
            input = Input(shape=input_shape)
            lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
            lstm1 = Dropout(0.5)(lstm1)
            lstm2 = Bidirectional(LSTM(32))(lstm1)
            lstm2 = Dropout(0.5)(lstm2)

        return Model(input, lstm2)

    def model_create(self):
        '''搭建网络'''
        embedding_layer = Embedding(input_dim=self.__VOCAB_SIZE,
                                    output_dim=self.__EMBEDDING_DIM,
                                    weights=[self.__EMBEDDING_MATRIX],
                                    input_length=self.__SEQUENCE_LENGTH,
                                    trainable=False,
                                    mask_zero=False)
        left_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='float32', name="left_x")
        right_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='float32', name='right_x')
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = self.__create_base_network(input_shape=(self.__SEQUENCE_LENGTH, self.__EMBEDDING_DIM))
        # shared_lstm.supports_masking = True

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # distance = Lambda(lambda x: self.__exponent_neg_manhattan_distance(x[0], x[1]),
        #                   output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        distance = Lambda(self.__exponent_neg_manhattan_distance2)([left_output, right_output])
        model = Model(inputs=[left_input, right_input],
                      outputs=distance)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        model.summary()
        return model