import numpy as np
from math import sqrt
from sklearn.model_selection import StratifiedKFold
from PlantDictionary import PlantDictionary
from neuron import mlp_ma_3w


def transposition(array):
    arrayT = np.empty((len(array[0]), len(array)), dtype=int)
    for i in range(len(array)):
        for j in range(len(array[i])):
            arrayT[j][i] = array[i][j]

    return arrayT

# wstępna obróbka danych
dane = []

file = open('dataset/data_Mar_64.txt', 'r')
for line in file.readlines():
    dane.append(line.split(','))

file.close()

# okreslenie ilosci wierszy oraz kolumn
numberOfRows = len(dane)
numberOfColumns = len(dane[0])

# podzial danych z transpozycja
data = np.empty((numberOfRows, numberOfColumns-1))
target = np.empty((numberOfRows, 1), dtype=int)
i = 0
j = 1

for wiersz in dane:
    target[i][0] = PlantDictionary[wiersz.pop(0)]
    data[i] = wiersz
    i += 1

# algorytm
# data = transposition(data)

max_epoch = 1000
display_freq = 10
err_goal = 0.05
lr = 0.000001
mc = 0.90
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04

K1 = 10
K3 = 1
K2 = int(sqrt(K1*K3))

samples_per_leaf = 16
train_test_rate = 0.8

train_sample_range = int(samples_per_leaf * train_test_rate)

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(0, len(data)-1, samples_per_leaf):
    x_train.extend(data[i:i + train_sample_range])
    y_train.extend(target[i:i + train_sample_range])
    x_test.extend(data[i:i + samples_per_leaf - train_sample_range])
    y_test.extend(target[i:i + samples_per_leaf - train_sample_range])

x_trainT = transposition(x_train)
y_trainT = transposition(y_train)
x_testT = transposition(x_test)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)

x_trainT = np.array(x_trainT)
y_trainT = np.array(y_trainT)
x_testT = np.array(x_testT)


mlpnet = mlp_ma_3w(x_trainT, y_train, K1, K2, lr, err_goal, display_freq, mc, ksi_inc, ksi_dec, er, max_epoch)
mlpnet.train(x_trainT, y_trainT)
result = mlpnet.predict(x_testT)

PK = (1 - sum((abs(result - y_test)>=0.5).astype(int)[0])/1200 ) * 100

print("Test #{:<2}: PK {} test_size {}".format(i, PK, 1200))


