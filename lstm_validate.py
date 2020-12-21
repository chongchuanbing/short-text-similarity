# -*- coding:utf-8 -*-

from core.lstm import Lstm

lstm = Lstm()

validate_file_path = './data/mytest_pair.csv'

# lstm.validate(validate_file_path, batch_size=1024)

# lstm.validate_1_N('./data/knowledge_question.csv', batch_size=512)

lstm.validate_1_N('./data/atec_question.csv', batch_size=512)

# lstm.print_w2v(validate_file_path)
