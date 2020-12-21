# -*- coding:utf-8 -*-

import os
import random

from core.lstm import Lstm

lstm = Lstm()

def get_enhancement():

    base_dir = './data/enhancement'
    file_path_arr = []

    enhancement_arr = os.listdir(base_dir)

    for enhancement_item in enhancement_arr:
        if '.DS_Store' != enhancement_item:
            file_path_arr.append(os.path.join(base_dir, enhancement_item))

    return file_path_arr

enhancement_file_path_arr = get_enhancement()

if len(enhancement_file_path_arr) > 0:
    enhancement_file_path_arr = random.sample(enhancement_file_path_arr, 1000)

print('enhancement_file count: %d' % (len(enhancement_file_path_arr)))

train_file_path_arr = [
    # './data/train.csv',
    './data/train_atec.csv',
    # './data/data_enhancement.csv',
    # './data/data_word_enhancement.csv'
    ]

lstm.train(train_file_path_arr,
           enhancement_file_path_arr,
           epochs=20,
           batch_size=512,
           validation_split=0.3,
           enhancement_split=0)
