import datetime
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pyarrow.compute import scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class Forecaster:

    def __init__(self):
        print("Loading data...")
        self.price_df = pd.read_parquet("./data/^SPX.parquet")
        print("Data loading complete")
        start_test_date = "2025-01-01"
        # self.train_data = price_df.loc[:start_test_date]
        # self.test_data = price_df.loc[start_test_date:]

        # TODO temp
        # self.train_data = price_df.loc[:]

    def run_LSTMv2(self, n_lookback = 30, n_forecast = 53):

        # Splits data into test and training set
        train_data = self.price_df.iloc[:-n_forecast]
        test_data = self.price_df.iloc[-n_forecast:]
        print("Training")
        print(train_data)
        print("Testing")
        print(test_data)

        # Scales the data using Min Max
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)

        # Generates inputs and outputs based on lookback and forcasting periods

        x_train = []
        y_train = []
        for i in range(n_lookback, len(scaled_train) - n_forecast + 1):
            x_train.append(scaled_train[i - n_lookback: i])
            y_train.append(scaled_train[i])

        # Convert lists to numpy arrays so that Input layer works
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Creates LSTM model
        model = Sequential()
        model.add(Input(shape=(n_lookback, 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)


        # # Initialize model
        # model = Sequential()
        # model.add(Input(shape=(n_lookback, 1)))
        # # LSTM layer 1
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 2
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 3
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 4
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 5
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 6
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 7
        # model.add(LSTM(units=50))
        # model.add(Dropout(0.20))
        # # final layer
        # model.add(Dense(1))
        # model.summary()
        #
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)


        # Generates predictions on test set using last training data
        # x_pred = scaled_train[-n_lookback - n_forecast:-n_forecast] # Gets enough data to predict last window of training data
        # x_pred = scaled_train[-n_lookback:] # Gets enough data to predict last window of training data
        y_pred_list = []
        for num_pred in range(n_forecast):
            new_window = pd.concat([self.price_df.iloc[-n_lookback - n_forecast + num_pred:-n_forecast]['Price'], pd.Series(y_pred_list[-num_pred:])], ignore_index=True)
            new_window = new_window.to_numpy().reshape(-1, 1)
            print(f"Old length: {len(self.price_df.iloc[-n_lookback - n_forecast + num_pred:-n_forecast]['Price'])}")
            print(f"New length: {len(pd.Series(y_pred_list[-num_pred:]))}")
            x_pred = scaler.transform(pd.DataFrame(new_window))
            y_pred_list.append(scaler.inverse_transform(model.predict(x_pred.reshape(1, n_lookback, 1))).flatten()[0])
        y_pred_df = pd.DataFrame(y_pred_list, columns=['Price'])
        print("y_pred_df")
        print(y_pred_df)

        return

        # plot all the series together
        #TODO I do not think these lines are connecting properly
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(train_data.index, train_data['Price'], label='Training data')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(test_data.index, y_pred, color='orange', label='Predicted Stock Price')

        plt.title('SPX Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('SPX Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        #report performance
        mse = mean_squared_error(test_data['Price'], y_pred)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data['Price'], y_pred)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data['Price'], y_pred))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(y_pred - test_data['Price']) / np.abs(test_data['Price']))
        print('MAPE: ' + str(mape))

        # plt.figure(figsize=(12, 6))
        # plt.plot(self.train_data.loc[self.train_data.index <= '2020-11-18']['Price'], 'b', label="Original Price")
        # plt.plot(self.train_data.loc[self.train_data.index <= '2020-11-18'].index, y_pred, 'r', label="Predicted Price")
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Organize the results in a dataframe for printing
        df_past = pd.DataFrame(train_data)
        df_past.rename(columns={'Price': 'Actual'}, inplace=True)
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1] # Ensures data is connected
        print(df_past)

        df_future = pd.DataFrame(columns=['Actual', 'Forecast'])
        df_future['Forecast'] = y_pred
        df_future['Actual'] = np.nan
        df_future.index = pd.date_range(start=df_past.index[-1] + pd.Timedelta(days=1), periods=n_forecast)  # set index
        print(df_future)

        results = pd.concat([df_past, df_future])
        print(results)

        # plot the results
        results.plot(title='SPX Forecast')
        plt.show()


    def run_LSTM(self, n_lookback = 30, n_forecast = 53):

        # Splits data into test and training set
        train_data = self.price_df.iloc[:-n_forecast]
        test_data = self.price_df.iloc[-n_forecast:]
        print("Training")
        print(train_data)
        print("Testing")
        print(test_data)

        # Scales the data using Min Max
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)

        # Generates inputs and outputs based on lookback and forcasting periods

        x_train = []
        y_train = []
        for i in range(n_lookback, len(scaled_train) - n_forecast + 1):
            x_train.append(scaled_train[i - n_lookback: i])
            y_train.append(scaled_train[i: i + n_forecast])

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
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
        """
        MSE: 18081.942519307137
        MAE: 120.39972330729167
        RMSE: 134.46911362579564
        MAPE: 0.020008558931045105
        maybe with epoc 10
        """


        # # Initialize model
        # model = Sequential()
        # model.add(Input(shape=(n_lookback, 1)))
        # # LSTM layer 1
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 2
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 3
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 4
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 5
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 6
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.20))
        # # LSTM layer 7
        # model.add(LSTM(units=50))
        # model.add(Dropout(0.20))
        # # final layer
        # model.add(Dense(n_forecast))
        # model.summary()
        #
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
        """
        MSE: 7435.416274722417
        MAE: 66.07750651041667
        RMSE: 86.2288598714051
        MAPE: 0.01106870861961337
        """


        # Generates predictions on test set using last training data
        # x_pred = scaled_train[-n_lookback - n_forecast:-n_forecast] # Gets enough data to predict last window of training data
        # x_pred = scaled_train[-n_lookback:] # Gets enough data to predict last window of training data
        print("Trying to predict with")
        print(self.price_df.iloc[-n_lookback - n_forecast:-n_forecast])
        x_pred = scaler.transform(self.price_df.iloc[-n_lookback - n_forecast:-n_forecast])
        y_pred = scaler.inverse_transform(model.predict(x_pred.reshape(1, n_lookback, 1))).flatten()
        y_pred_df = pd.DataFrame(y_pred, columns=['Price'])
        print("y_pred_df")
        print(y_pred_df)

        # plot all the series together
        #TODO I do not think these lines are connecting properly
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(train_data.index, train_data['Price'], label='Training data')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(test_data.index, y_pred, color='orange', label='Predicted Stock Price')

        plt.title('SPX Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('SPX Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        #report performance
        mse = mean_squared_error(test_data['Price'], y_pred)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data['Price'], y_pred)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data['Price'], y_pred))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(y_pred - test_data['Price']) / np.abs(test_data['Price']))
        print('MAPE: ' + str(mape))

        # plt.figure(figsize=(12, 6))
        # plt.plot(self.train_data.loc[self.train_data.index <= '2020-11-18']['Price'], 'b', label="Original Price")
        # plt.plot(self.train_data.loc[self.train_data.index <= '2020-11-18'].index, y_pred, 'r', label="Predicted Price")
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Organize the results in a dataframe for printing
        df_past = pd.DataFrame(train_data)
        df_past.rename(columns={'Price': 'Actual'}, inplace=True)
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1] # Ensures data is connected
        print(df_past)

        df_future = pd.DataFrame(columns=['Actual', 'Forecast'])
        df_future['Forecast'] = y_pred
        df_future['Actual'] = np.nan
        df_future.index = pd.date_range(start=df_past.index[-1] + pd.Timedelta(days=1), periods=n_forecast)  # set index
        print(df_future)

        results = pd.concat([df_past, df_future])
        print(results)

        # plot the results
        results.plot(title='SPX Forecast')
        plt.show()