# -*- coding:utf-8 -*-

import codecs
import os
import jieba

from keras.preprocessing.text import Tokenizer

import _pickle as cPickle



DATA_DIR = './data/'
TRAIN_DATA_FILE = DATA_DIR + 'character.txt'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000

save_path = "./Params"
tokenizer_name = "character_tokenizer.pkl"


words = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    line_content = f.readline()
    while line_content:

        words.append([char for char in line_content if char])

        line_content = f.readline()

print('Found %s texts in train.csv' % len(words))

print("Fit tokenizer...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(words)
print("Save tokenizer...")
if not os.path.exists(save_path):
    os.makedirs(save_path)
cPickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))

