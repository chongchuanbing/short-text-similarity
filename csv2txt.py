

import pandas as pd

df_item = pd.read_csv('./data/train_atec.csv', header=None)

with open('./data/train_atec.txt', 'w') as f:
    for index, row in df_item.iterrows():
        q1 = row[0]
        q2 = row[1]
        score = row[2]

        f.write(q1 + '\t' + q2 + '\t' + str(score) + '\n')