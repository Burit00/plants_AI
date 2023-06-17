import pandas as pd
import numpy as np

datas = pd.read_csv('dataset/data_Mar_64_prepare.txt', index_col='spec')

data = datas.iloc[:, :]

mins = np.min(data)
maxs = np.max(data)

range = maxs - mins
data = (data - mins)/range

pd.DataFrame.to_csv(data, 'dataset/data_Mar_64_norm.txt')
