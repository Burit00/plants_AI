import numpy as np
from math import sqrt
from sklearn.model_selection import StratifiedKFold
from PlantDictionary import PlantDictionary
from neuron import mlp_ma_3w

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
wejscie = np.empty((numberOfRows, numberOfColumns-1))
wyjscie = np.empty((numberOfRows, 1), dtype=int)
i = 0
j = 1

for wiersz in dane:
    wyjscie[i][0] = PlantDictionary[wiersz.pop(0)]
    wejscie[i] = wiersz
    i += 1

data = wejscie
target = wyjscie

max_epoch = 100
err_goal = 0.1
disp_freq = 10
lr = 0.00001
mc = 0.90
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04

K1 = 10
K3 = 1
K2 = int(sqrt(K1*K3))

CVN = 2
skfold = StratifiedKFold(n_splits=CVN, shuffle=True)
PK_vec = np.zeros(CVN)

for i, (train, test) in enumerate(skfold.split(data, np.squeeze(target)), start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]

    mlpnet = mlp_ma_3w(x_train.T, y_train, K1, K2, lr, err_goal, disp_freq, mc, ksi_inc, ksi_dec, er, max_epoch)
    mlpnet.train(x_train.T, y_train.T)
    result = mlpnet.predict(x_test.T)
    n_test_samples = test.size
    PK_vec[i] = (1 - sum((abs(result - y_test) >= 0.5).astype(int)[0])/n_test_samples) * 100

    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))

PK = np.mean(PK_vec)
print("PK {}".format(PK))
