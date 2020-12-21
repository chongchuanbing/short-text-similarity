# -*- coding:utf-8 -*-


from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Lambda, CuDNNLSTM, Conv1D, MaxPool1D, Concatenate, MaxPooling1D, GlobalMaxPool1D, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.preprocessing.text
from keras import backend as K


def lstm_layer(q, lstm1, lstm2):
    q = lstm1(q)
    q = Dropout(0.3)(q)
    q = lstm2(q)
    q = Lambda(lambda x: K.reshape(x, (-1, 30, 256)))(q)
    return q


def conv_pool(conv_unit, q):
    q_conv = conv_unit(q)
    q_maxp = MaxPool1D(pool_size=30)(q_conv)
    q_maxp = Lambda(lambda x: K.reshape(x, (-1, int(x.shape[-1]))))(q_maxp)
    q_meanp = Lambda(lambda x: K.mean(x, axis=1))(q_conv)
    return q_maxp, q_meanp


def mix_layer(q1_maxp, q1_meanp, q2_maxp, q2_meanp):
    add_q_max = Lambda(lambda x: x[0] + x[1])([q1_maxp, q2_maxp])
    sub_q_max = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_maxp, q2_maxp])
    mul_q_max = concatenate([q1_maxp, q2_maxp])
    square_max = Lambda(lambda x: K.square(x[0] - x[1]))([q1_maxp, q2_maxp])

    add_q_mean = Lambda(lambda x: x[0] + x[1])([q1_meanp, q2_meanp])
    sub_q_mean = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_meanp, q2_meanp])
    mul_q_mean = concatenate([q1_meanp, q2_meanp])
    square_mean = Lambda(lambda x: K.square(x[0] - x[1]))([q1_meanp, q2_meanp])

    return Concatenate()([q1_maxp, q2_maxp, add_q_max, sub_q_max, mul_q_max, square_max,
                          q1_meanp, q2_meanp, add_q_mean, sub_q_mean, mul_q_mean, square_mean])


class Multi_lstm_cnn_model(object):

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
        # 构建embedding层，q1 和 q2共享此embedding层
        embedding_layer = Embedding(input_dim=self.__VOCAB_SIZE,
                                    output_dim=self.__EMBEDDING_DIM,
                                    weights=[self.__EMBEDDING_MATRIX],
                                    input_length=self.__SEQUENCE_LENGTH,
                                    trainable=False)
        # 词嵌入
        sequence_1_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='int32')
        embed_1 = embedding_layer(sequence_1_input)
        sequence_2_input = Input(shape=(self.__SEQUENCE_LENGTH,), dtype='int32')
        embed_2 = embedding_layer(sequence_2_input)
        # lstm
        lstm_layer_1 = CuDNNLSTM(256, return_sequences=True)
        lstm_layer_2 = CuDNNLSTM(256, return_sequences=True)
        # lstm_layer_1 = LSTM(256, return_sequences=True)
        # lstm_layer_2 = LSTM(256, return_sequences=True)
        q1 = lstm_layer(embed_1, lstm_layer_1, lstm_layer_2)
        q2 = lstm_layer(embed_2, lstm_layer_1, lstm_layer_2)
        # 用类似TextCNN的思路构建不同卷积核的特征，两个句子共用同样的卷积层
        kernel_size = [2, 3, 4, 5]
        conv_concat = []
        for kernel in kernel_size:
            conv = Conv1D(64, kernel_size=kernel, activation='relu', padding='same')
            q1_maxp, q1_meanp = conv_pool(conv, q1)
            q2_maxp, q2_meanp = conv_pool(conv, q2)
            mix = mix_layer(q1_maxp, q1_meanp, q2_maxp, q2_meanp)
            conv_concat.append(mix)
        conv = Concatenate()(conv_concat)
        # 全连接层
        merged = Dropout(0.3)(conv)
        merged = BatchNormalization()(merged)
        merged = Dense(512, activation='relu', name='dense_output')(merged)

        # merged = Dropout(0.3)(merged)
        # merged = BatchNormalization()(merged)
        # merged = Dense(256, activation='relu', name='dense_output2')(merged)

        # merged = Dropout(0.3)(merged)
        # merged = BatchNormalization()(merged)
        # merged = Dense(128, activation='relu', name='dense_output3')(merged)

        merged = Dropout(0.3)(merged)
        merged = BatchNormalization(name='bn_output')(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        return model