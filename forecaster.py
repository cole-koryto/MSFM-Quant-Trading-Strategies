import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pyarrow.compute import scalar
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class Forecaster:

    def __init__(self):
        price_df = pd.read_parquet("./data/^SPX.parquet")
        start_test_date = "2025-01-01"
        self.train_data = price_df.loc[:start_test_date]
        self.test_data = price_df.loc[start_test_date:]


    def run_LSTM(self):
        # Scales the data using Min Max
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(self.train_data)
        scaled_train = scaler.transform(self.train_data)

        # Generates inputs and outputs based on lookback and forcasting periods
        n_lookback = 30
        n_forecast = 30
        x_train = []
        y_train = []
        for i in range(n_lookback, len(scaled_train) - n_forecast + 1):
            x_train.append(scaled_train[i - n_lookback: i])
            y_train.append(scaled_train[i: i + n_lookback])

        # Convert lists to numpy arrays so that Input layer works
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Creates LSTM model
        model = Sequential()
        model.add(Input(shape=(n_lookback, 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)

        # Generates predictions on test set using last training data
        x_pred = scaled_train[-n_lookback:]
        y_pred = scaler.inverse_transform(model.predict(x_pred.reshape(1, n_lookback, 1)))

        # Organize the results in a dataframe for printing
        df_past = pd.DataFrame(self.train_data)
        df_past.rename(columns={'Price': 'Actual'}, inplace=True)
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1] # Ensures data is connected
        print(df_past)

        df_future = pd.DataFrame(columns=['Actual', 'Forecast'])
        df_future['Forecast'] = y_pred.flatten()
        df_future['Actual'] = np.nan
        df_future.index = pd.date_range(start=df_past.index[-1] + pd.Timedelta(days=1), periods=n_forecast)  # set index
        print(df_future)

        results = pd.concat([df_past, df_future])
        print(results)

        # plot the results
        results.plot(title='SPX Forecast')
        plt.show()