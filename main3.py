import pandas as pd
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(learning_rate, momentum, draw_loss_curve=False, hidden_layer_sizes=(100, 100)):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam',
                          learning_rate='adaptive',
                          learning_rate_init=learning_rate,
                          momentum=momentum, max_iter=15000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność modelu: {:.2f}%".format(accuracy * 100))
    if draw_loss_curve:
        plt.plot(model.loss_curve_)
        plt.title("Krzywa uczenia")
        plt.xlabel("Liczba epok")
        plt.ylabel("Funkcja kosztu")
        plt.show()

    return accuracy


def eksperyment1():
    momentum_list = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    lr_list = []
    accuracy_score_list = []

    for lr_p in range(5):
        lr = 10 ** -(lr_p+1)

        lr_list.append(lr)
        accuracy_score_list.append([])
        for momentum in momentum_list:
            accuracy = train(lr, momentum)
            accuracy_score_list[lr_p].append(accuracy)

    accuracy_score_list = np.array(accuracy_score_list)
    lr_list = np.array(lr_list)
    momentum_list = np.array(momentum_list)
    hkl.dump([lr_list, momentum_list, accuracy_score_list], 'eksperyment1.hkl')


def eksperyment2():
    first_layer_size_list = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    second_layer_size_list = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    accuracy_score_list = []
    for i, first_layer_size in enumerate(first_layer_size_list):
        accuracy_score_list.append([])
        for j, second_layer_size in enumerate(second_layer_size_list):
            accuracy = train(0.001, 0.9, hidden_layer_sizes=(first_layer_size, second_layer_size))
            accuracy_score_list[i].append(accuracy)

    accuracy_score_list = np.array(accuracy_score_list)

    hkl.dump([first_layer_size_list, second_layer_size_list, accuracy_score_list], 'eksperyment2.hkl')


data = pd.read_csv('dataset/data_Mar_64.txt')

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# train(0.001, 0.9, draw_loss_curve=True)
print('Eksperyment #1')
eksperyment1()
print('Eksperyment #2')
eksperyment2()
