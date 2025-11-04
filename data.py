import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


data = pd.read_csv('GOOG.csv',date_parser=True)
data_training = data[data['Date']<'2018-01-01'].copy()
data_test = data[data['Date']>='2018-01-01'].copy()
training_data = data_training.drop(['Date','Adj Close'],axis = 1)
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
X_train=[]
y_train=[]




for i in range(60,training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    y_train.append(training_data[i,0])

X_train = np.array(X_train)



y_train = np.array(y_train)


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()