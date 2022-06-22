import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
st.set_page_config(layout="wide")


def download_stocks_history(stksybl):
    stock_final = pd.DataFrame()
    stock_symbol = stksybl 
    try:
        stock = []
        stock = yf.download(stock_symbol,period='5y')
        if len(stock) == 0:
            None
        else:
            stock_final = stock_final.append(stock,sort=False)
    except Exception:
        None
    return stock_final

def load_model():
    model = keras.models.load_model('finalized_model.h5')
    #st.write(model)
    return model

def predict_future_data(next_prediction_days,stocks_data):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_reference_data = 60 
    next_future_days = next_prediction_days
    LSTM_test_data = np.log(stocks_data[int(len(stocks_data)*0.9):])
  
    x_input=LSTM_test_data[len(LSTM_test_data)-past_reference_data:].values.reshape(-1,1)
    x_temp_input = scaler.fit_transform(x_input)
    x_input = x_temp_input.reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    from numpy import array
    from datetime import timedelta
    from datetime import date
    import matplotlib.pyplot as plt
    import time
    LSTM_model = load_model()
    lst_output=[]
    n_steps=past_reference_data
    i=0

    while(i<next_future_days):
        
        if(len(temp_input)>past_reference_data):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = LSTM_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = LSTM_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    day_new=np.arange(1,past_reference_data+1)
    day_pred=np.arange(past_reference_data+1,past_reference_data+1+next_future_days)

    # Predict Next Stock Price
    lstm_nxt30_stocks = np.exp(scaler.inverse_transform(lst_output))
    future_30days_stock_predictions = pd.DataFrame(index=range(0,next_future_days),columns=['Date', 'Prediction'])
    Begindate = date.today()
    for i in range(0,next_future_days):
        future_30days_stock_predictions['Date'][i] = Begindate + timedelta(days=i+1)
        future_30days_stock_predictions['Prediction'][i] = lstm_nxt30_stocks[i][0]
        
    with st.spinner('Wait for it...'):
        time.sleep(5)  

    future_30days_stock_predictions.set_index('Date',inplace=True) 
    
    global col1, col2
    col1, col2 = st.columns(2)
       
    
    with col1:
        st.dataframe(future_30days_stock_predictions,width=500,height=2000)

    # Plot Next Stock Price
    with col2:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        ax.plot(day_new,np.exp(scaler.inverse_transform(x_temp_input[len(x_temp_input)-past_reference_data:])),color = 'blue',label='Past data')
        ax.plot(day_pred,future_30days_stock_predictions,color = 'orange',label='Next '+str(next_prediction_days)+' days Data')
        ax.legend()
        st.pyplot(fig)
    return future_30days_stock_predictions
 
 


def main():
    import time
    st.title("Stock Price Prediction")
    predictions = []
    stk_option = st.selectbox(
    'Choose Your Stock',
    ('TCS', 'DMART'),)
    predicted_days_option = st.selectbox(
    'No Of Future Days?',
    (30,1,2,3,4,5,10,25,50,60))
    st.write('You selected : '+str(stk_option)+' For prediction of next '+str(predicted_days_option)+' days')
    if st.button('Predict'):
        companyCode = stk_option
        next_prediction_days = predicted_days_option
        stocks_data = download_stocks_history(companyCode+".NS") 
        predictions = predict_future_data(next_prediction_days,stocks_data["Adj Close"])

        
             
    
if __name__=='__main__':
    main()



