# -*- coding:utf-8 -*-


from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

import pandas as pd
import numpy as np
import abc

class Base_data_deal(metaclass = abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def parse_sentence(self, sentence):
        pass

    def load_data(self,
                  file_path_arr,
                  validation_split
                  ):
        '''
        解析csv，解析txt等
        :return:
        '''

        df = pd.DataFrame(columns=['q1', 'q2', 'label'])
        for file_path_item in file_path_arr:
            df_item = pd.read_csv(file_path_item, header=None)
            df_item.columns = ['q1', 'q2', 'label']

            df = df.append(df_item)

        df_0 = df.loc[df['label'] == 0, :]
        df_1 = df.loc[df['label'] == 1, :]

        df_0_count = df_0['label'].count()
        df_1_count = df_1['label'].count()

        min_count = min(df_0_count, df_1_count)

        df_0_frac = min_count * (validation_split / 2) / df_0_count
        df_1_frac = min_count * (validation_split / 2) / df_1_count

        # df_0_val = df_0.sample(frac=df_0_frac)
        # df_1_val = df_1.sample(frac=df_1_frac)

        df_0_train, df_0_val = train_test_split(df_0, test_size=df_0_frac, shuffle=True, random_state=42)
        df_1_train, df_1_val = train_test_split(df_1, test_size=df_1_frac, shuffle=True, random_state=42)

        df_train = df_0_train.append(df_1_train)
        df_val = df_0_val.append(df_1_val)

        df_train = df_train.sample(frac=1)
        df_val = df_val.sample(frac=1)

        return df_train, df_val

    def load_enhancement(self,
                           df_train,
                           enhancement_file_path_arr,
                           enhancement_split):
        if 0 == enhancement_split:
            return df_train
        df = pd.DataFrame(columns=['q1', 'q2', 'label'])
        for file_path_item in enhancement_file_path_arr:
            df_item = pd.read_csv(file_path_item, header=None)
            df_item.columns = ['q1', 'q2', 'label']

            df = df.append(df_item)

        df_split = df.sample(frac=enhancement_split)

        print('enhancemant count: %d' % (df_split.q1.count()))

        return df_train.append(df_split)
    
    def load_embedding(self,
                       embedding_path,
                       embedding_dim):
        embeddings_dict = {}
        with open(embedding_path, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < embedding_dim:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs

        return embeddings_dict

    def build_vocab(self,
                    df
                    ):
        '''
        计算词集
        :return:
        '''
        vocabs = {'UNK'}

        texts_1 = df['q1'].apply(lambda x: self.parse_sentence(x)).tolist()
        texts_2 = df['q2'].apply(lambda x: self.parse_sentence(x)).tolist()

        for sentence_item in texts_1:
            for item in sentence_item:
                vocabs.add(item)
        for sentence_item in texts_2:
            for item in sentence_item:
                vocabs.add(item)

        return vocabs

    def word2vec_dictionaries2(self,
                                 vocabs_all,
                                 embedding_path,
                                 embedding_dim
                                 ):

        embeddings_dict = self.load_embedding(embedding_path, embedding_dim)

        # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
        word_index = {wd:(index) for index, wd in enumerate(list(vocabs_all))}
        word_vector = {}  # 初始化`[word : vector]`字典

        # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
        # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
        embeddings_matrix = np.zeros((len(vocabs_all) + 1, embedding_dim))

        ## 填充 上述 的字典 和 大矩阵
        for word, i in word_index.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                # 词语：词向量
                word_vector[word] = embedding_vector
                # 词向量矩阵
                embeddings_matrix[i + 1] = embedding_vector

        return word_index, word_vector, embeddings_matrix

    def data_transform2(self,
                          word_index,
                          df,
                        sequence_length):
        '''
        word2vec词向量编码索引
        :param word_index:
        :param df:
        :return:
        '''
        texts_1 = df['q1'].apply(lambda x: self.parse_sentence(x)).tolist()
        texts_2 = df['q2'].apply(lambda x: self.parse_sentence(x)).tolist()
        labels = df['label'].tolist()

        data_1 = self.word2vec_tokenizer(word_index, texts_1, sequence_length)
        data_2 = self.word2vec_tokenizer(word_index, texts_2, sequence_length)
        labels = np.array(labels)

        return data_1, data_2, labels

    def word2vec_tokenizer(self,
                             word_index,
                             texts,
                             sequence_length):
        '''
        文本转word2vec词向量索引
        :param word_index:
        :param texts:
        :return:
        '''
        data = []
        for sentence in texts:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(word_index[word])  # 把句子中的 词语转化为index
                except Exception as e:
                    print(e)
                    # new_txt.append(0)
                    pass

            data.append(new_txt)

        # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
        texts = sequence.pad_sequences(data, maxlen=sequence_length)

        return texts

    def encode(self,
               vocabs,
               word_index):
        '''
        编码
        :return:
        '''

        pass

    def build_embeddings_matrix(self):
        '''
        构建词向量矩阵
        :return:
        '''
        pass