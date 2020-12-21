# -*- coding:utf-8 -*-


from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Lambda, CuDNNLSTM, Conv1D, MaxPool1D, Concatenate, MaxPooling1D, GlobalMaxPool1D, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.preprocessing.text

class Simple_lstm_cnn_model(object):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 sequence_length,
                 embedding_matrix
                 ):
        self.__VOCAB_SIZE = vocab_size
        self.__EMBEDDING_DIM = embedding_dim
        self.__SEQUENCE_LENGTH = sequence_length
        self.__EMBEDDING_MATRIX = embedding_matrix

    def model_create(self):
        embedding_layer = Embedding(input_dim=self.__VOCAB_SIZE,
                                    output_dim=self.__EMBEDDING_DIM,
                                    weights=[self.__EMBEDDING_MATRIX],
                                    input_length=self.__SEQUENCE_LENGTH,
                                    trainable=False)

        lstm_layer = LSTM(128, dropout=0.15, recurrent_dropout=0.15)

        sequence_1_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)

        q1 = Conv1D(128, 5, activation='relu')(embedded_sequences_1)
        q1 = MaxPooling1D(pool_size=5, strides=1)(q1)
        q1 = Conv1D(128, 5, activation='relu')(q1)
        q1 = MaxPooling1D(pool_size=5, strides=1)(q1)
        # q1 = Conv1D(128, 5, activation='relu')(q1)
        # global max pooling
        # q1 = GlobalMaxPool1D()(q1)
        # q1 = Flatten()(q1)
        # q1 = Dense(128, activation='relu')(q1)

        sequence_2_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)

        q2 = Conv1D(128, 5, activation='relu')(embedded_sequences_2)
        q2 = MaxPooling1D(pool_size=5, strides=1)(q2)
        q2 = Conv1D(128, 5, activation='relu')(q2)
        q2 = MaxPooling1D(pool_size=5, strides=1)(q2)
        # q2 = Conv1D(128, 5, activation='relu')(q2)
        # global max pooling
        # q2 = GlobalMaxPool1D()(q2)
        # q2 = Flatten()(q2)
        # q2 = Dense(128, activation='relu')(q2)

        q1 = lstm_layer(q1)
        q1 = Dropout(0.3)(q1)
        q1 = BatchNormalization()(q1)
        q1 = Dense(200, activation='relu')(q1)

        q2 = lstm_layer(q2)
        q2 = Dropout(0.3)(q2)
        q2 = BatchNormalization()(q2)
        q2 = Dense(200, activation='relu')(q2)

        merged = concatenate([q1, q2])
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(512, activation='relu')(merged)

        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(256, activation='relu')(merged)

        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(128, activation='relu')(merged)

        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input], \
                      outputs=preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['acc'])
        model.summary()
        return model