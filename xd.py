import pandas as pd

data = pd.read_csv('dataset/data_Mar_64.txt')

X = data.iloc[:, 1:]
y = data.loc[0, 0]

print(y)