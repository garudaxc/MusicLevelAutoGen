import tensorflow
import keras
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def fx(x):
    y = (np.sin(x * 0.4)) * 0.5
    return y



# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df






X = [fx(i) for i in range(36)]
# df = DataFrame(X[0:10])
# df2 = df.shift(1)

# # print(df)
# # print(df2)
# d = [df, df.shift(1), df.shift(2)]
# d = concat(d, axis=1)
# d = d.values
# print(d)


plt.plot(X)
plt.show()