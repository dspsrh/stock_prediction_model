import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pickle


data = pd.read_csv('GOOG.csv',date_parser=True)
data_training = data[data['Date']<'2018-01-01'].copy()
data_test = data[data['Date']>='2018-01-01'].copy()
training_data = data_training.drop(['Date','Adj Close'],axis = 1)
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
pastData = data_training.tail(60)
df = pastData.append(data_test,ignore_index = True)
df = df.drop(['Date','Adj Close'], axis = 1)
inputs = scaler.transform(df)

regression = tensorflow.keras.models.load_model("GOOG-TanH-1st.model")


X_test = []
y_test=[]



for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i,0])

X_test,y_test=np.array(X_test),np.array(y_test)



y_pred=regression.predict(X_test)

scale = 1/scaler.scale_[0]

y_pred *= scale
y_test *= scale


plt.plot(y_test,color = 'red', label='Real Google Stock prive')
plt.plot(y_pred, color = 'blue', label='Predicted Google Stock Price')

plt.title('Google stock prediction with 1st model')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()
