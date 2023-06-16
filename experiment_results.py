import random

import hickle as hkl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


x = [10**-i for i in range(5)]
y = [random.random() * 10 for i in range(5)]
z = np.array([[i for i in range(5)] for j in range(5)])

ax = plt.axes(projection='3d')
x, y = np.meshgrid(x, y)

print(x, y, z)
ax.plot_surface(x, y, z, cmap=cm.viridis, antialiased=False)
ax.set_xscale('logit')
plt.show()

# eksperyment 1
lr_list, momentum_list, accuracy_score_list = hkl.load('eksperyment1.hkl')


ax = plt.axes(projection='3d')
momentum_list, lr_list = np.meshgrid(momentum_list, lr_list)
ax.plot_surface(lr_list, momentum_list, accuracy_score_list, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('3d')
ax.set_xlabel('learning rate')
ax.set_xscale('logit', linthresh=0.01)
ax.set_ylabel('momentum')
ax.set_zlabel('accuracy')
plt.show()


# eksperyment 2
first_layer_size_list, second_layer_size_list, accuracy_score_list = hkl.load('eksperyment2.hkl')


ax = plt.axes(projection='3d')
first_layer_size_list, second_layer_size_list = np.meshgrid(first_layer_size_list, second_layer_size_list)
ax.plot_surface(first_layer_size_list, second_layer_size_list, accuracy_score_list, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('Zależność dokładności modelu od liczby neuronów')
ax.set_xlabel('Liczba neuronów 1 warstwy')
ax.set_ylabel('Liczba neuronów 2 warstwy')
ax.set_zlabel('Dokładność modelu')
plt.show()
