import numpy as np
import hickle as hkl

from nnet_leaves import *


def calc_SSE(array):
    SSE = 0.0
    for el in array:
        SSE += el**2
    return SSE

def experiment_1():
    momentum_list = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    lr_list_to_file = np.array(lr_list)
    momentum_list_to_file = np.array(momentum_list)
    accuracy_score_list = []

    for i, lr in enumerate(lr_list):
        accuracy_score_list.append([])

        for mc in momentum_list:
            accuracy = train(learning_rate=lr, momentum=mc)
            accuracy_score_list[i].append(accuracy)
            print('lr', lr)
            print('mc', mc)


        accuracy_score_list_to_file = np.array(accuracy_score_list)
        hkl.dump([lr_list_to_file, momentum_list_to_file, accuracy_score_list_to_file], 'eksperyment1.hkl')


def experiment_2():
    first_layer_size_list = np.array([i for i in range(5, 105, 5)])
    second_layer_size_list = np.array([i for i in range(5, 105, 5)])
    accuracy_score_list = []

    for i, first_layer_size in enumerate(first_layer_size_list):
        accuracy_score_list.append([])
        for j, second_layer_size in enumerate(second_layer_size_list):
            accuracy = train(hidden_layer_sizes=(first_layer_size, second_layer_size))
            accuracy_score_list[i].append(accuracy)

        accuracy_score_list_to_file = np.array(accuracy_score_list)
        hkl.dump([first_layer_size_list, second_layer_size_list, accuracy_score_list_to_file], 'eksperyment_2.hkl')


def experiment_3():
    lr_inc_values = [1.1, 1.2, 1.3, 1.4, 1.5]
    lr_dec_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    er = 1.04
    max_epoch = 1000

    accuracy_score_list = np.zeros((len(lr_inc_values), len(lr_dec_values)))
    sse_list = np.zeros((len(lr_inc_values), len(lr_dec_values)))

    for i, lr_inc in enumerate(lr_inc_values):
        for j, lr_dec in enumerate(lr_dec_values):
            model = create_model()

            learning_rate = model.learning_rate_init
            accuracy = 0.0
            sse = 0.0
            sse_prev = 0.0
            y_pred = 0

            for epoch in range(1, max_epoch+1):
                model.partial_fit(X_train, y_train, classes=np.unique(y))

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                sse_prev = sse
                sse = calc_SSE(y_pred - y_test)
                if sse > er * sse_prev:
                    learning_rate *= lr_dec
                elif sse < er * sse_prev:
                    learning_rate *= lr_inc

                model.set_params(learning_rate_init=learning_rate)

            accuracy_score_list[i, j] = accuracy

            sse = ((y_pred - y_test) ** 2).sum()
            sse_list[i, j] = sse

        accuracy_score_list_to_file = np.array(accuracy_score_list)
        sse_list_to_file = np.array(sse_list)
        hkl.dump([lr_inc_values, lr_dec_values, accuracy_score_list_to_file, sse_list_to_file], 'eksperyment3.hkl')
