#%%
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd

#------------------------------------------------------------
#拿前幾年的股價 去預測下一年的股價
#------------------------------------------------------------
def run():
    dat_config = 60

    dataset_train2106 = pd.read_csv('data/TWSE2016_en.csv')
    dataset_train2107 = pd.read_csv('data/TWSE2017_en.csv')
    dataset_train = pd.concat((dataset_train2106, dataset_train2107), axis = 0)
    dataset_train = dataset_train[dataset_train['stock'] == '2317 鴻海']
    dataset_train['date'] = pd.to_datetime(dataset_train['date'])
    dataset_train.sort_values('date', inplace=True)
    #training_set = dataset_train.iloc[:, 2:3] #open
    dataset_train_list = dataset_train.iloc[:, 5:6].values #close
    training_set = pd.DataFrame(dataset_train_list)


    #%%
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    #%%
    X_train = []
    y_train = []
    for i in range(dat_config,len(training_set_scaled) ):
        X_train.append(training_set_scaled[i-dat_config:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #%%
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 5, batch_size = 64)


    #%%
    #test
    dataset_test = pd.read_csv('data/TWSE2018_en.csv')
    dataset_test = dataset_test[dataset_test['stock'] == '2317 鴻海']
    dataset_test['date'] = pd.to_datetime(dataset_test['date'])
    dataset_test.sort_values('date', inplace=True)
    #real_stock_price = dataset_test.iloc[:, 2:3].values #open
    real_stock_price = dataset_test.iloc[:, 5:6].values #close

    #%%
    dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0) #open
    dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis = 0) #close
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - dat_config:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs)
    X_test = []
    a = len(inputs)
    for i in range(dat_config, a):
        X_test.append(inputs[i-dat_config:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    real_stock_price = real_stock_price[:,0]
    real_stock_price = [float(i) for i in real_stock_price]
    real_stock_price = np.around(real_stock_price,1)

    predicted_stock_price = predicted_stock_price[:,0]
    predicted_stock_price = [float(i) for i in predicted_stock_price]
    predicted_stock_price = np.around(predicted_stock_price,1)
    #%%
    plt.plot(real_stock_price, color = 'red', label = '2317 Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted 2317 Stock Price')
    plt.title('2317 remuneration Prediction')
    plt.xlabel('Time')
    plt.ylabel('remuneration')
    plt.yticks(np.arange(70,140, 5))
    plt.legend()
    plt.show()