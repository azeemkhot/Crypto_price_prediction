import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import date

import pickle

# from google.colab import files
# uploaded = files.upload()

# data = pd.read_csv(io.BytesIO(uploaded['BTC-USD (1).csv'])) 


start='2014-01-01'
end=date.today()

rnn_mape,lstm_mape=[],[]

cryptocurrenices=['BTC','ETH','SOL','DOGE','MATIC','LTC','XRP','SHIB','TRX','BNB']

for crypto in cryptocurrenices:

    data=yf.download(crypto+'-USD',start,end)
    data.reset_index(inplace=True)
    print(data)

    """
    <a id="2"></a>
    ## 2.Data Preprocessing
    """

    data.info()

    data.fillna(method='ffill',inplace=True)
    data.info()

    data['Mean']=data.iloc[:,1:5].mean(axis=1)

    data

    """
    <a id="2"></a>
    ## 3.Spliting Data as Train and Test
    """

    length_data = len(data)    
    split_ratio = 0.8           # %80 train + %20 test
    length_train = round(length_data * split_ratio)  
    length_test = length_data - length_train
    print("Data length :", length_data)
    print("Train data length :", length_train)
    print("Test data lenth :", length_test)

    train_data = data[:length_train].iloc[:,[0,7]] 
    train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object
    train_data

    test_data = data[length_train:].iloc[:,[0,7]]
    test_data['Date'] = pd.to_datetime(test_data['Date'])  # converting to date time object
    test_data

    """
    <a id="3"></a>
    ## 4.Creating Train Dataset from Train split
    """

    dataset_train = train_data.Mean.values
    dataset_train.shape

    # Change 1d array to 2d array
    # Changing shape from (1692,) to (1692,1)
    dataset_train = np.reshape(dataset_train, (-1,1))
    dataset_train.shape

    """
    #### <a id="4"></a>
    ## 5.Normalization
    """

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()


    # scaling dataset
    dataset_train_scaled = scaler.fit_transform(dataset_train)

    dataset_train_scaled
    dataset_train_scaled.shape

    plt.subplots(figsize = (15,6))
    plt.plot(dataset_train_scaled)
    plt.xlabel("Days")
    plt.ylabel(crypto+" Normalized Mean Price")
    plt.show()

    """
    <a id="5"></a>
    ## 6.Creating X_train and y_train from Train data
    """

    X_train = []
    y_train = []

    time_step = 50

    for i in range(time_step, length_train):
        X_train.append(dataset_train_scaled[i-time_step:i,0])
        y_train.append(dataset_train_scaled[i,0])
        
    # convert list to array
    X_train, y_train = np.array(X_train), np.array(y_train)

    print("Shape of X_train before reshape :",X_train.shape)
    print("Shape of y_train before reshape :",y_train.shape)

    """
    ## Reshape
    """

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))

    print("Shape of X_train after reshape :",X_train.shape)
    print("Shape of y_train after reshape :",y_train.shape)

    X_train[0]

    y_train[0]

    """
    <a id="6"></a>
    ## 7.Creating RNN model 
    """

    # importing libraries
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import SimpleRNN
    from keras.layers import Dropout

    # initializing the RNN
    regressor = Sequential()

    # adding first RNN layer and dropout regulatization
    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True, 
                input_shape = (X_train.shape[1],1))
                )

    regressor.add(
        Dropout(0.2)
                )


    # adding second RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
                )

    regressor.add(
        Dropout(0.2)
                )

    # adding third RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
                )

    regressor.add(
        Dropout(0.2)
                )

    # adding fourth RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50)
                )

    regressor.add(
        Dropout(0.2)
                )

    # adding the output layer
    regressor.add(Dense(units = 1))

    # compiling RNN
    regressor.compile(
        optimizer = "adam", 
        loss = "mean_squared_error",
        metrics = ["accuracy"])

    # fitting the RNN
    history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
    regressor.save(crypto+'-RNN.h5')

    """
    <a id="8"></a>
    ## 8.Creating Test Dataset from Test Data 
    """

    dataset_test = test_data.Mean.values  
    dataset_test = np.reshape(dataset_test, (-1,1))  # converting 1D to 2D array
    scaled_dataset_test =  scaler.fit_transform(dataset_test)  # scaling open values to between 0 and 1
    print("Shape of scaled test dataset :",scaled_dataset_test.shape)

    # Creating X_test and y_test
    X_test = []
    y_test = []

    for i in range(time_step, length_test):
        X_test.append(scaled_dataset_test[i-time_step:i,0])
        y_test.append(scaled_dataset_test[i,0])

    # Converting to array
    X_test, y_test = np.array(X_test), np.array(y_test)

    print("Shape of X_test before reshape :",X_test.shape)
    print("Shape of y_test before reshape :",y_test.shape)

    """
    ### Reshape
    """

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array
    y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array

    print("Shape of X_test after reshape :",X_test.shape)
    print("Shape of y_test after reshape :",y_test.shape)

    """
    <a id="9"></a>
    ## 9.Evaluating RNN with Test Data 
    """

    # predictions with X_test data
    y_pred_of_test = regressor.predict(X_test)
    # scaling back from 0-1 to original
    y_pred_of_test = scaler.inverse_transform(y_pred_of_test) 
    print("Shape of y_pred_of_test :",y_pred_of_test.shape)

    # visualisation
    plt.figure(figsize = (15,6))
    plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
    plt.plot(scaler.inverse_transform(y_test), label = "y_test", c = "g")
    plt.xlabel("Days")
    plt.ylabel("Mean price")
    plt.title("Simple RNN model - "+crypto+", Prediction with input X_test vs y_test")
    plt.legend()
    plt.show()


    from sklearn.metrics import mean_absolute_percentage_error

    rnn_mape.append(round(mean_absolute_percentage_error(scaler.inverse_transform(y_test), y_pred_of_test)*100,2))


    """
    <a id="10"></a>
    ## 10.Creating LSTM Model
    """

    from keras.layers import LSTM

    model_lstm = Sequential()
    model_lstm.add(
        LSTM(50,return_sequences=True,input_shape = (X_train.shape[1],1)))
    model_lstm.add(
        LSTM(50, return_sequences= False))
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
    history2 = model_lstm.fit(X_train, y_train, epochs = 25, batch_size = 16)
    model_lstm.save(crypto+'-LSTM.h5')

    """
    <a id="11"></a>
    ## 11.Evaluating LSTM Model with test data
    """

    y_pred_of_test=model_lstm.predict(X_test)
    y_pred_of_test=scaler.inverse_transform(y_pred_of_test)

    plt.subplots(figsize =(15,6))
    plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
    plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")
    plt.xlabel("Days")
    plt.ylabel("Mean price")
    plt.title("LSTM model - "+crypto+", Predictions with input X_test vs y_test")
    plt.legend()
    plt.show()


    lstm_mape.append(round(mean_absolute_percentage_error(scaler.inverse_transform(y_test), y_pred_of_test)*100,2))





  
X_axis = np.arange(len(cryptocurrenices))
  
plt.bar(X_axis - 0.15, rnn_mape, 0.3, label = 'RNN')
plt.bar(X_axis + 0.15, lstm_mape, 0.3, label = 'LSTM')
  
plt.xticks(X_axis, cryptocurrenices)
plt.xlabel("Cryptocurrencies")
plt.ylabel("MAPE in %")
plt.title("MAPE for RNN and LSTM for all cryptocurrencies")
plt.legend()
plt.savefig('all-crypto-accuracy.png')
plt.show()


rnn_mape_final=sum(rnn_mape)/len(cryptocurrenices)
lstm_mape_final=sum(lstm_mape)/len(cryptocurrenices)

plt.bar(['RNN','LSTM'], [rnn_mape_final,lstm_mape_final])
plt.xlabel("Algorithm used")
plt.ylabel("MAPE in %")
plt.title("MAPE of the Algorithms")
plt.savefig('algorithm-accuracy.png')
plt.show()

























'''
dataset = data.iloc[:,[0,7]] 
dataset['Date'] = pd.to_datetime(dataset['Date'])  # converting to date time object
print(dataset)


"""
<a id="3"></a>
## 4.Creating Train Dataset from Train split
"""

dataset = dataset.Mean.values
dataset.shape

# Change 1d array to 2d array
# Changing shape from (1692,) to (1692,1)
dataset = np.reshape(data, (-1,1))
dataset.shape

"""
#### <a id="4"></a>
## 5.Normalization
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# scaling dataset
dataset_scaled = scaler.fit_transform(dataset)

dataset_scaled
dataset_scaled.shape

plt.subplots(figsize = (15,6))
plt.plot(dataset_scaled)
plt.xlabel("Days")
plt.ylabel("Normalized Mean Price")
plt.show()

"""
<a id="5"></a>
## 6.Creating X_train and y_train from Train data
"""

X_train = []
y_train = []

for i in range(time_step, length_train):
    X_train.append(dataset_scaled[i-time_step:i,0])
    y_train.append(dataset_scaled[i,0])
    
# convert list to array
X_train, y_train = np.array(X_train), np.array(y_train)

print("Shape of X_train before reshape :",X_train.shape)
print("Shape of y_train before reshape :",y_train.shape)

"""
## Reshape
"""

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))

print("Shape of X_train after reshape :",X_train.shape)
print("Shape of y_train after reshape :",y_train.shape)

X_train[0]

y_train[0]


regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
pickle.dump(regressor,open('BTC-rnn','wb'))

model_lstm.fit(X_train, y_train, epochs = 10, batch_size = 10)
pickle.dump(model_lstm,open('BTC-lstm','wb'))
'''