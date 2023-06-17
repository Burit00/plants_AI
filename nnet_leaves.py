import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_model(learning_rate=0.01, momentum=0.9, hidden_layer_sizes=(30, 80)):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam',
                          learning_rate='adaptive',
                          learning_rate_init=learning_rate,
                          momentum=momentum, max_iter=10000)
    return model


def train(learning_rate=0.01, momentum=0.9, hidden_layer_sizes=(50, 30), draw_loss_curve=False):
    model = create_model(learning_rate=learning_rate, momentum=momentum, hidden_layer_sizes=hidden_layer_sizes)

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


data = pd.read_csv('dataset/data_Mar_64_norm.txt', index_col='spec')


data = data.iloc[:, :]

X_train, X_test, y_train, y_test = train_test_split(data, data.index.values, test_size=0.2, random_state=42)

# strat = X_train.groupby(['spec']).count()

#train(0.005, 0.9, draw_loss_curve=True)
#print('Eksperyment #1')
#eksperyment1()
#print('Eksperyment #2')
#eksperyment2()
