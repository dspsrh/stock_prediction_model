from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pickle

X_train = pickle.load(open("X.pickle","rb"))
y_train = pickle.load(open("y.pickle","rb"))
regression = Sequential()

regression.add(LSTM(units = 50,return_sequences=True,input_shape = (X_train.shape[1],5)))
regression.add(Dropout(0.2))

regression.add(LSTM(units = 60,return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units = 80,return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units = 120))
regression.add(Dropout(0.2))


regression.add(Dense(units = 1))


regression.compile(optimizer='adam',loss = 'mean_squared_error')


regression.fit(X_train,y_train, epochs = 100, batch_size = 32)


regression.save('GOOG-TanH-1st.model')