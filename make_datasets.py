import pandas as pd
import random
from collections import defaultdict, Counter
import os

df = pd.read_csv("/home/samuelchen/overall_database.csv")
index = defaultdict(list)
for idx, i in enumerate(df['hosts']):
    index[i].append(idx)

imbalance_class_train_count = 40
random.seed(88)

if imbalance_class_train_count > 50:
    imbalance_class_train_count = 50
q = str(imbalance_class_train_count)
val_num = test_num = 16



for type in ["standard", "expend"]:
    train, val, test = [], [], []

    for value in index.values():
        random.shuffle(value)
        val.extend(value[:val_num])
        test.extend((value[val_num:val_num + test_num]))
        if type == "standard" and imbalance_class_train_count < 50:
            train.extend(value[val_num + test_num:val_num + test_num + imbalance_class_train_count])
        else:
            if len(value) <= 100:
                train.extend(value[val_num + test_num:val_num + test_num + imbalance_class_train_count])
            else:
                train.extend(value[val_num + test_num:])


    X_train = df['sequences'].loc[train]
    X_val = df['sequences'].loc[val]
    X_test = df['sequences'].loc[test]
    Y_train = df['hosts'].loc[train]
    Y_val = df['hosts'].loc[val]
    Y_test = df['hosts'].loc[test]


    aa = [(0, 'Artibeus lituratus'), (1, 'Bos taurus'), (2, 'Canis lupus'), (3, 'Capra hircus'), (4, 'Cerdocyon thous'), (5, 'Desmodus rotundus'), (6, 'Eptesicus fuscus'), (7, 'Equus caballus'), (8, 'Felis catus'), (9, 'Homo sapiens'), (10, 'Lasiurus borealis'), (11, 'Mephitis mephitis'), (12, 'Nyctereutes procyonoides'), (13, 'Procyon lotor'), (14, 'Tadarida brasiliensis'), (15, 'Vulpes lagopus'), (16, 'Vulpes vulpes')]
    dic = defaultdict(int)
    dicc = defaultdict(str)
    for i, j in aa:
        dic[j] = i
        dicc[i] = j
    print(dicc)

    count = Counter(Y_train)
    count = sorted(count.items(), key=lambda x:x[1])
    mn = count[-1][1]
    ans = []
    for i in count:
        ans.append(str(dic[i[0]]) + ':{:.2f}'.format(mn / i[1]))
    print(count)
    print(ans)
    outpath = "/home/samuelchen/" + q + "/" + type
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    X_train.to_csv(outpath + '/X_train.csv', sep='\t', encoding='utf-8', header=False)
    X_val.to_csv(outpath + '/X_val.csv', sep='\t', encoding='utf-8', header=False)
    X_test.to_csv(outpath + '/X_test.csv', sep='\t', encoding='utf-8', header=False)
    Y_train.to_csv(outpath + '/Y_train.csv', sep='\t', encoding='utf-8', header=False)
    Y_val.to_csv(outpath + '/Y_val.csv', sep='\t', encoding='utf-8', header=False)
    Y_test.to_csv(outpath + '/Y_test.csv', sep='\t', encoding='utf-8', header=False)
