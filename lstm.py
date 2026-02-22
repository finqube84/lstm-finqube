import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def run():
    st.title("📈 LSTM Stock Price Prediction Dashboard")

    ticker_symbol = st.text_input("Enter Stock Ticker (e.g. RELIANCE.NS)", "RELIANCE.NS")

    if st.button("Run Prediction"):
        # Download Data
        stock_data = yf.download(ticker_symbol, start="2015-01-01")
        stock_data = stock_data[['Close']]
        stock_data.dropna(inplace=True)

        data = stock_data['Close'].values.reshape(-1,1)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data)*0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(dataset, time_step=20):
            X,y = [],[]
            for i in range(len(dataset)-time_step-1):
                X.append(dataset[i:(i+time_step),0])
                y.append(dataset[i+time_step,0])
            return np.array(X), np.array(y)

        time_step = 20

        X_train,y_train = create_dataset(train_data,time_step)
        X_test,y_test = create_dataset(test_data,time_step)

        X_train = X_train.reshape(X_train.shape[0],time_step,1)
        X_test = X_test.reshape(X_test.shape[0],time_step,1)

        # Build Model
        model = Sequential([
            Input(shape=(time_step, 1)),
            LSTM(16),
            Dense(1)
        ])
        model.compile(loss='mse', optimizer='adam')

        # Model Training
        with st.spinner('Training the LSTM model... please wait.'):
            model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=0)

        # Predictions
        train_predict = scaler.inverse_transform(model.predict(X_train))
        test_predict = scaler.inverse_transform(model.predict(X_test))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        mae = mean_absolute_error(y_test_actual, test_predict)
        
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")

        # Plot Actual vs Predicted
        st.subheader("📈 Test Data: Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(y_test_actual, label="Actual Price", color='blue')
        ax1.plot(test_predict, label="Predicted Price", color='orange')
        ax1.legend()
        st.pyplot(fig1)

        # Future 100 Day Prediction
        last_sequence = scaled_data[-time_step:]
        current_batch = last_sequence.reshape(1,time_step,1)

        future_days = 100
        future_predictions = []

        for _ in range(future_days):
            pred = model.predict(current_batch, verbose=0)[0]
            future_predictions.append(pred)
            current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)

        future_predictions = scaler.inverse_transform(future_predictions)
        future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
        future_df = pd.DataFrame(future_predictions, index=future_dates, columns=["Predicted"])

        st.subheader("🔮 Next 100 Days Forecast")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(stock_data['Close'].tail(500), label="Recent Historical")
        ax2.plot(future_df, label="Forecast", color='red')
        ax2.legend()
        st.pyplot(fig2)

        st.dataframe(future_df)

# EXTREMELY IMPORTANT: No spaces before 'if', and exactly 4 spaces before 'run()'
if __name__ == "__main__":
    run()
