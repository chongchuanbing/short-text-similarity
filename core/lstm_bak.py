# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import csv
import codecs
# import matplotlib.pyplot as plt

import numpy as np

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Lambda, CuDNNLSTM
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.preprocessing.text

import _pickle as cPickle

from keras import backend as K

# from tools import jieba_util


import jieba

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./Params/stop_words_ch.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence, remove_stopwords=False):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip(), cut_all=False)

    if remove_stopwords:
        # 创建一个停用词列表
        stopwords = stopwordslist()
        # 输出结果为outstr

        word_arr = []
        for word in sentence_depart:
            if word not in stopwords:
                if word != '\t':
                    word_arr.append(word)
        return ' '.join(word_arr)

    return ' '.join(sentence_depart)

########################################
# set directories and parameters
########################################
DATA_DIR = './data/'
EMBEDDING_FILE = './Params/w2v.mod'
EMBEDDING_FILE = './Params/w2v.mod'
TRAIN_DATA_FILE = DATA_DIR + 'mytrain_pair.csv'
TRAIN_DATA_FILE = DATA_DIR + 'train.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

# num_lstm = np.random.randint(175, 275)
# num_dense = np.random.randint(100, 150)
# rate_drop_lstm = 0.15 + np.random.rand() * 0.25
# rate_drop_dense = 0.15 + np.random.rand() * 0.25

num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = './Params/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                                rate_drop_dense)

save = True
load_tokenizer = False
save_path = "./Params"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./Params/embedding_matrix.npy"

# ########################################
# ## index word vectors
# ########################################
# print('Indexing word vectors')
#
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
# process texts in datasets
########################################
print('Processing text dataset')


texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    for values in reader:
        texts_1.append(seg_depart(values[0]))
        texts_2.append(seg_depart(values[1]))
        labels.append(int(values[2]))
print('Found %s texts in train.csv' % len(texts_1))


'''
this part is solve keras.preprocessing.text can not process unicode
start here
'''


def text_to_word_sequence(text,
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


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
'''
end here
'''

print('Load tokenizer...')
tokenizer = cPickle.load(open(os.path.join(save_path, tokenizer_name), 'rb'))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
# print sequences_1

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of datadeal tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)


########################################
# prepare embeddings
########################################
print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)

        # print(word, embedding_matrix[i])
    else:
        print(word)

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

np.save(embedding_matrix_path, embedding_matrix)


# #######################################
# # sample train/validation datadeal
# #######################################
# np.random.seed(1234)
# perm = np.random.permutation(len(data_1))
# idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
# idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]
#
# data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
# data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
# labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
#
# data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
# data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
# labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

# weight_val = np.ones(len(labels_val))
# if re_weight:
#     weight_val *= 0.472001959
#     weight_val[labels_val == 0] = 1.309028344


########################################
# define the model structure
########################################
def get_model():
    """
    可以通过weights参数指定初始的embedding
    因为Embedding层是不可导的
    梯度东流至此回,所以把embedding放在中间层是没有意义的,emebedding只能作为第一层
    注意weights到embeddings的绑定过程很复杂，weights是一个列表
    """
    embedding_layer = Embedding(input_dim=nb_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation='relu')(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input], \
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model



def create_base_network(input_shape):
    '''搭建编码层网络,用于权重共享'''

    input = Input(shape=input_shape)
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input)
    lstm1 = Dropout(0.5)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(32))(lstm1)
    lstm2 = Dropout(0.5)(lstm2)

    return Model(input, lstm2)


def exponent_neg_manhattan_distance(sent_left, sent_right):
    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))


def bilstm_siamese_model():
    '''搭建网络'''
    embedding_layer = Embedding(input_dim=nb_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False,
                                mask_zero=False)
    left_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name="left_x")
    right_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name='right_x')
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)
    shared_lstm = create_base_network(input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    # shared_lstm.supports_masking = True

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                      output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    model = Model(inputs=[left_input, right_input],
                  outputs=distance)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    return model

#######################################
# train the model
########################################

# def draw_train(history):
#     '''绘制训练曲线'''
#     # Plot training & validation accuracy values
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
#
#     # Plot training & validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.savefig("model/result_atec.png")
#     plt.show()


def train_model():
    print(STAMP)

    # model = get_model()

    model = bilstm_siamese_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'

    # 'model - ep{epoch: 03d} - loss{loss: .3f} - val_loss{val_loss: .3f}.h5'

    model_checkpoint = ModelCheckpoint(
            bst_model_path,
            # monitor='val_acc',
            monitor='val_loss',
            mode='max',
            period=1,
            save_best_only=True,
            save_weights_only=True)

    tensorboard = TensorBoard(log_dir='logs')

    # hist = model.fit([data_1, data_2], labels, \
    #                  validation_data=([data_1, data_2], labels), \
    #                  epochs=100, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    hist = model.fit(
        # x=[data_1_train, data_2_train],
        # y=labels_train,
        # validation_data=([data_1_val, data_2_val], labels_val),

        x=[data_1, data_2],
        y=labels,
        validation_split=VALIDATION_SPLIT,

        epochs=100,
        batch_size=512,
        shuffle=True,
        # callbacks=[early_stopping, model_checkpoint],
        callbacks=[model_checkpoint, tensorboard]
        # callbacks = [tensorboard]
    )

    model.load_weights(bst_model_path)

    # draw_train(hist)

    bst_score = min(hist.history['loss'])
    bst_acc = max(hist.history['acc'])
    print(bst_acc, bst_score)


if __name__ == '__main__':
    train_model()

# predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)

# for i in range(count):
#    print "t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i])
