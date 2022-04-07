import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import yfinance as yf
from datetime import date
from plotly import graph_objs as go
from keras.models import load_model



moduleNames=['BTC','ETH','SOL','DOGE','MATIC','LTC','XRP','SHIB','TRX','BNB']




st.title('Cryptocurrency price prediction')

cryptos=('BTC','ETH','SOL','DOGE','MATIC','LTC','XRP','SHIB','TRX','BNB')
selected_crypto=st.selectbox('Select Cryptocurrency for prediction',cryptos)

print(selected_crypto)
model_rnn=load_model(selected_crypto+'-RNN.h5')
model_lstm=load_model(selected_crypto+'-LSTM.h5')

start='2014-01-01'
end=date.today()

df=yf.download(selected_crypto+'-USD',start,end)
df.reset_index(inplace=True)



df.fillna(method='ffill',inplace=True)
df.info()

df['Mean']=df.iloc[:,1:5].mean(axis=1)



st.header('Dataset for selected crypto')
st.write(df)

fig=go.Figure()
fig.add_trace(go.Scatter(x=df['Date'],y=df['Mean'],name='Mean Price'))
fig.layout.update(title_text='Historical Data', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


length_data = len(df)    
split_ratio = 0.8           # %80 train + %20 test
length_train = round(length_data * split_ratio)  
length_test = length_data - length_train
print("Data length :", length_data)
print("Train data length :", length_train)
print("Test data lenth :", length_test)

test_data = df[length_train:].iloc[:,[0,7]]
test_data['Date'] = pd.to_datetime(test_data['Date'])  # converting to date time object


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


dataset_test = test_data.Mean.values  
dataset_test = np.reshape(dataset_test, (-1,1))  # converting 1D to 2D array
scaled_dataset_test =  scaler.fit_transform(dataset_test)  # scaling open values to between 0 and 1


# Creating X_test and y_test
X_test = []
y_test = []

time_step = 50

for i in range(time_step, length_test):
    X_test.append(scaled_dataset_test[i-time_step:i,0])
    y_test.append(scaled_dataset_test[i,0])

# Converting to array
X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array
y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array

st.header('Output for Test Dataset')
st.subheader('RNN output')

y_pred_of_test = model_rnn.predict(X_test)
# scaling back from 0-1 to original
y_pred_of_test = scaler.inverse_transform(y_pred_of_test) 

rnnfig=plt.figure(figsize = (15,6))
plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
plt.plot(scaler.inverse_transform(y_test), label = "y_test", c = "g")
plt.xlabel("Days")
plt.ylabel("Mean price")
plt.title("Simple RNN model - "+selected_crypto+", Prediction with input X_test vs y_test")
plt.legend()
st.pyplot(rnnfig)

st.subheader('LSTM output')

y_pred_of_test=model_lstm.predict(X_test)
y_pred_of_test=scaler.inverse_transform(y_pred_of_test)

lstmfig=plt.figure(figsize =(15,6))
plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")
plt.xlabel("Days")
plt.ylabel("Mean price")
plt.title("LSTM model - "+selected_crypto+", Predictions with input X_test vs y_test")
plt.legend()
st.pyplot(lstmfig)


#Tomorrows prediction
st.header("Future prediction")

X_input = df.iloc[-time_step:].Mean.values               # getting last 50 rows and converting to array
X_input = scaler.fit_transform(X_input.reshape(-1,1))      # converting to 2D array and scaling
X_input = np.reshape(X_input, (1,50,1))                    # reshaping : converting to 3D array



simple_RNN_prediction = scaler.inverse_transform(model_rnn.predict(X_input))
LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input))
print("Simple RNN, Mean price prediction      :", simple_RNN_prediction[0,0])
print("LSTM prediction, Mean price prediction :", LSTM_prediction[0,0])


col1, col2 = st.columns(2)
col1.metric("RNN", "$"+str(simple_RNN_prediction[0,0]),str(round(((simple_RNN_prediction[0,0]-df.iloc[-1].Mean)/df.iloc[-1].Mean)*100,2))+"%")
col2.metric("LSTM", "$"+str(LSTM_prediction[0,0]), str(round(((LSTM_prediction[0,0]-df.iloc[-1].Mean)/df.iloc[-1].Mean)*100,2))+"%")