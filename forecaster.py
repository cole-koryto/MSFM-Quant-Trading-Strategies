import datetime
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyarrow.compute import scalar
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Forecaster:

    def __init__(self):
        print("Loading data...")
        self.price_df = pd.read_parquet("./data/ES=F.parquet")
        self.price_df["return"] = self.price_df["Price"].pct_change()
        print("Data loading complete")
        start_test_date = "2025-01-01"
        # self.train_data = price_df.loc[:start_test_date]
        # self.test_data = price_df.loc[start_test_date:]

        # TODO temp
        # self.train_data = price_df.loc[:]

    def run_LSTMv2(self, n_lookback = 30, n_forecast = 53):

        # Splits data into test and training set
        # train_data = self.price_df[["Price"]].iloc[:-n_forecast].dropna()
        # test_data = self.price_df[["Price"]].iloc[-n_forecast:].dropna()
        train_data = self.price_df[["return"]].iloc[:-n_forecast].dropna()
        test_data = self.price_df[["return"]].iloc[-n_forecast:].dropna()
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
            y_train.append(scaled_train[i, 0])

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
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=False, verbose=2)


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
        # model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=False, verbose=2)


        # Generates predictions on test set using last training data
        # x_pred = scaled_train[-n_lookback - n_forecast:-n_forecast] # Gets enough data to predict last window of training data
        # x_pred = scaled_train[-n_lookback:] # Gets enough data to predict last window of training data
        # Prepare input: last `n_lookback` points from train_data
        lookback = train_data[-n_lookback:].copy()  # Series or array

        # Scale initial input
        input_seq = scaler.transform(np.array(lookback).reshape(-1, 1))  # shape: (n_lookback, 1)

        # Initialize prediction list
        predictions = []

        for _ in range(n_forecast):
            # Reshape input for LSTM: (1 sample, n_lookback steps, 1 feature)
            x_input = input_seq.reshape(1, n_lookback, 1)

            # Predict next return
            y_pred = model.predict(x_input, verbose=0)[0][0]  # extract scalar

            # Append prediction to list
            predictions.append(y_pred)

            # Update input_seq: remove first row, append prediction (as a row)
            input_seq = np.append(input_seq[1:], [[y_pred]], axis=0)

        # Convert predictions to original scale
        y_pred_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Ensure test_data is an array
        true_values = np.array(test_data[:len(predictions)])  # in case predictions < test_data length

        # Report performance
        print(y_pred_rescaled)
        mse = mean_squared_error(true_values, y_pred_rescaled)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(true_values, y_pred_rescaled)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(true_values, y_pred_rescaled))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(y_pred_rescaled - true_values) / np.abs(true_values))
        print('MAPE: ' + str(mape))

        # plot all the series together
        # TODO I do not think these lines are connecting properly
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(train_data.index[-30:], train_data.iloc[-30:]['return'], label='Training data')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(test_data.index, y_pred_rescaled, color='orange', label='Predicted Stock Price')

        plt.title('SPX Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('SPX Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        return

        y_pred_list = []
        current_input = x_train[-1]
        print("Now predicting")
        print(current_input)
        for num_pred in range(n_forecast):
            pred = model.predict(current_input)
            print(pred)
            y_pred_list.append(scaler.inverse_transform(pred))
            print(y_pred_list)

            current_input = current_input[1:] + [pred]


            # new_window = pd.concat([scaled_train[-n_lookback - n_forecast:], pd.Series(y_pred_list[-min(num_pred,n_lookback):])], ignore_index=True)
            # new_window = new_window.to_numpy().reshape(-1, 1)
            # x_pred = scaler.transform(pd.DataFrame(new_window))
            # y_pred_list.append(scaler.inverse_transform(model.predict(x_pred.reshape(1, n_lookback, 1))).flatten()[0])
        y_pred_df = pd.DataFrame(y_pred_list, columns=['Price'])
        print("y_pred_df")
        print(y_pred_df)


        # plot all the series together
        #TODO I do not think these lines are connecting properly
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(train_data.index, train_data['Price'], label='Training data')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(test_data.index, y_pred_df['Price'], color='orange', label='Predicted Stock Price')

        plt.title('SPX Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('SPX Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        #report performance
        mse = mean_squared_error(test_data['Price'], y_pred_df['Price'])
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data['Price'], y_pred_df['Price'])
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data['Price'], y_pred_df['Price']))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(y_pred_df['Price'] - test_data['Price']) / np.abs(test_data['Price']))
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
        df_future['Forecast'] = y_pred_df['Price']
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
        train_data = self.price_df[["Price"]].iloc[:-n_forecast].dropna()
        test_data = self.price_df[["Price"]].iloc[-n_forecast:].dropna()
        print("Training")
        print(train_data)
        print("Testing")
        print(test_data)

        # Scales the data using Min Max
        # scaler = StandardScaler()
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

        # # Creates LSTM model
        # model = Sequential()
        # model.add(Input(shape=(n_lookback, 1)))
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(LSTM(units=50))
        # model.add(Dense(n_forecast))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
        """
        MSE: 18081.942519307137
        MAE: 120.39972330729167
        RMSE: 134.46911362579564
        MAPE: 0.020008558931045105
        maybe with epoc 10
        """


        # Initialize model
        model = Sequential()
        model.add(Input(shape=(n_lookback, 1)))
        # LSTM layer 1
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 2
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 3
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 4
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 5
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 6
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.20))
        # LSTM layer 7
        model.add(LSTM(units=50))
        model.add(Dropout(0.20))
        # final layer
        model.add(Dense(n_forecast))
        model.summary()

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=False, verbose=2)
        """
        MSE: 7435.416274722417
        MAE: 66.07750651041667
        RMSE: 86.2288598714051
        MAPE: 0.01106870861961337
        """


        # Generates predictions on test set using last training data
        # x_pred = scaled_train[-n_lookback - n_forecast:-n_forecast] # Gets enough data to predict last window of training data
        # x_pred = scaled_train[-n_lookback:] # Gets enough data to predict last window of training data
        # x_pred = scaler.transform(self.price_df.iloc[-n_lookback - n_forecast:-n_forecast])
        # x_pred = scaler.transform(train_data[-n_lookback:])
        x_pred_seq = scaled_train[-n_lookback:].reshape(1, n_lookback, 1)
        print("Trying to predict with")
        print(x_pred_seq)
        y_pred = scaler.inverse_transform(model.predict(x_pred_seq)).flatten()
        y_pred_df = pd.DataFrame(y_pred, columns=['Price'])
        print("RAW")
        print(model.predict(x_pred_seq.reshape(1, n_lookback, 1)))
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
        plt.ylabel('SPX Return')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # report performance
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
        # df_past = pd.DataFrame(train_data)
        # df_past.rename(columns={'return': 'Actual'}, inplace=True)
        # df_past['Forecast'] = np.nan
        # df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1] # Ensures data is connected
        # # print(df_past)
        #
        # df_future = pd.DataFrame(columns=['Actual', 'Forecast'])
        # df_future['Forecast'] = y_pred
        # df_future['Actual'] = np.nan
        # df_future.index = pd.date_range(start=df_past.index[-1] + pd.Timedelta(days=1), periods=n_forecast)  # set index
        # # print(df_future)
        #
        # results = pd.concat([df_past, df_future])
        # # print(results)
        #
        # # plot the results
        # results.plot(title='SPX Forecast')
        # plt.show()

    def run_XGBoost(self):
        # Parameters
        n_forecast = 53  # number of days into the future to predict
        lags = 30  # number of lag days to use as features

        # Load stock data
        df = self.price_df

        # Create features and target
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['return'].shift(i)

        df[f'target_{n_forecast}d'] = df['return'].shift(-n_forecast)
        df.dropna(inplace=True)

        # Features and labels
        print(df)
        features = [f'lag_{i}' for i in range(1, lags + 1)]
        x = df[features]
        y = df[f'target_{n_forecast}d']

        # Train/test split
        x_train = x[:-n_forecast]
        x_test = x[-n_forecast:]
        y_train = y[:-n_forecast]
        y_test = y[-n_forecast:]

        # Model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(x_train, y_train)

        # Predictions
        y_pred = model.predict(x_test)

        # Report performance
        mse = mean_squared_error(y_test, y_pred)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(y_test, y_pred)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        print('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(y_pred - y_test) / np.abs(y_test))
        print('MAPE: ' + str(mape))

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[:-n_forecast], y_train, label='Training Actual')
        plt.plot(df.index[-n_forecast:], y_test, label='Testing Actual')
        plt.plot(df.index[-n_forecast:], y_pred, label='Predicted')
        plt.title(f'{n_forecast}-Day Ahead Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_XGBoostv2(self):
        # Parameters
        n_forecast = 180  # how many days ahead we want to forecast iteratively
        lags = 30 # how many past days to use as input

        # Creates modeling df
        df = self.price_df.copy()

        # Creates return quantiles
        df['quantile'] = pd.qcut(df['return'], q=5, labels=False)
        print(f"Quantile portions:\n{df['quantile'].value_counts(normalize=True).sort_index()}")

        # Create lag features
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['quantile'].shift(i)

        # One-day ahead target
        df['target_1d'] = df['quantile'].shift(-1)
        df.dropna(inplace=True)

        # Add 2-day, 3-day, and 5-day moving averages for return
        # df['MA_2_r'] = df['return'].rolling(window=2).mean()
        # df['MA_3_r'] = df['return'].rolling(window=3).mean()
        # df['MA_5_r'] = df['return'].rolling(window=5).mean()

        # Add 2-day, 3-day, and 5-day moving averages for return quintiles
        df['MA_2_q'] = df['quantile'].rolling(window=2).mean()
        df['MA_3_q'] = df['quantile'].rolling(window=3).mean()
        df['MA_5_q'] = df['quantile'].rolling(window=5).mean()

        # Features and target
        features = [f'lag_{i}' for i in range(1, lags + 1)]
        features.extend(['MA_2_q', 'MA_3_q', 'MA_5_q'])
        x = df[features]
        y = df['target_1d']

        # Train/test split
        x_train = x[:-n_forecast]
        x_test = x[-n_forecast:]
        y_train = y[:-n_forecast]
        y_test = y[-n_forecast:]

        # Create a TimeSeriesSplit object for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Train the model
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [5, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.4, 0.5],
            "colsample_bytree": [0.8],
            "min_child_weight": [5, 10],
            "gamma": [0.1, 0.2],
            "reg_lambda": [10, 15],
        }
        xgb = XGBClassifier(
            objective='multi:softprob', # provides actual class probabilities
            num_class=5,
            tree_method='hist',         # faster than approx
            eval_metric='mlogloss',     # good for multi-class
            random_state=25,
            n_jobs=-1
        )
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',     # could also do neg_log_loss if we focus on probability calibration
            verbose=2,
            n_jobs=-1)
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Predict on test set
        y_pred = model.predict(x_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Quantile Classifier)")
        plt.show()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-n_forecast-30:-n_forecast], y_train[-30:], label='Training Actual')
        plt.plot(df.index[-n_forecast:], y_test, label='Testing Actual')
        plt.plot(df.index[-n_forecast:], y_pred, label='Predicted')
        plt.title(f'{n_forecast}-Day Ahead Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()