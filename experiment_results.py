import hickle as hkl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# eksperyment 1
lr_list, momentum_list, accuracy_score_list = hkl.load('eksperyment1.hkl')

ax = plt.axes(projection='3d')
momentum_list, lr_list = np.meshgrid(momentum_list, lr_list)
ax.plot_surface(lr_list, momentum_list, accuracy_score_list, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('Zależność dokładności modelu od współczynnika uczenia i ')
ax.set_xlabel('learning rate')
ax.set_ylabel('momentum')
ax.set_zlabel('Dokładność modelu')
plt.show()


# eksperyment 2
first_layer_size_list, second_layer_size_list, accuracy_score_list = hkl.load('eksperyment_2.hkl')

ax = plt.axes(projection='3d')
first_layer_size_list, second_layer_size_list = np.meshgrid(first_layer_size_list, second_layer_size_list)
ax.plot_surface(first_layer_size_list, second_layer_size_list, accuracy_score_list, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('Zależność dokładności modelu od liczby neuronów')
ax.set_xlabel('Liczba neuronów 1 warstwy')
ax.set_ylabel('Liczba neuronów 2 warstwy')
ax.set_zlabel('Dokładność modelu')
plt.show()


# eksperyment 3
lr_inc_list, lr_dec_list, accuracy_score_list, sse_list = hkl.load('eksperyment3.hkl')

ax = plt.axes(projection='3d')
lr_inc_list, lr_dec_list = np.meshgrid(lr_inc_list, lr_dec_list)
ax.plot_surface(lr_inc_list, lr_dec_list, accuracy_score_list, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('Zależność dokładności modelu od współczynników \nmodyfikacji współczynnika uczenia')
ax.set_xlabel('lr_inc')
ax.set_ylabel('lr_dec')
ax.set_zlabel('Dokładność modelu')
plt.show()
