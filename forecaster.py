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
from tensorflow.keras.utils import to_categorical
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

    def run_LSTM(self):
        lags = 30  # number of lag features
        n_classes = 5  # number of quantiles
        n_forecast = 180

        df = self.price_df.copy()

        # Create return quantiles
        df['quantile'] = pd.qcut(df['return'], q=n_classes, labels=False)

        # One-day ahead target
        df['target_1d'] = df['quantile'].shift(-1)

        # Create lag features for quantiles
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['quantile'].shift(i)

        # Drop NaNs created by shifting
        df.dropna(inplace=True)

        # Features and target
        features = [f'lag_{i}' for i in range(1, lags + 1)]
        x = df[features].values
        y = df['target_1d'].values.astype(int)

        # -----------------------------
        # Train/Test Split
        # -----------------------------
        x_train = x[:-n_forecast]
        x_test = x[-n_forecast:]
        y_train = y[:-n_forecast]
        y_test = y[-n_forecast:]

        # -----------------------------
        # Scale features
        # -----------------------------
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Reshape for LSTM: (samples, timesteps=lags, features=1)
        x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], lags, 1))
        x_test_lstm = x_test_scaled.reshape((x_test_scaled.shape[0], lags, 1))

        # One-hot encode targets
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_test_cat = to_categorical(y_test, num_classes=n_classes)

        # -----------------------------
        # Build LSTM Model
        # -----------------------------
        model = Sequential()
        model.add(Input(shape=(lags, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(n_classes, activation='softmax'))

        """"
        Accuracy: 0.217

Classification Report:
              precision    recall  f1-score   support

           0      0.214     0.176     0.194        34
           1      0.109     0.132     0.119        38
           2      0.259     0.429     0.323        35
           3      0.263     0.128     0.172        39
           4      0.276     0.235     0.254        34

    accuracy                          0.217       180
   macro avg      0.224     0.220     0.212       180
weighted avg      0.223     0.217     0.210       180
        """

        # model = Sequential()
        # model.add(Input(shape=(lags, 1)))
        # model.add(LSTM(50, return_sequences=True))
        # model.add(Dropout(0.2))  # 20% dropout
        # model.add(LSTM(25))
        # model.add(Dropout(0.2))
        # model.add(Dense(n_classes, activation='softmax'))
        """
        ccuracy: 0.256

Classification Report:
              precision    recall  f1-score   support

           0      0.244     0.294     0.267        34
           1      0.238     0.132     0.169        38
           2      0.262     0.629     0.370        35
           3      0.400     0.051     0.091        39
           4      0.241     0.206     0.222        34

    accuracy                          0.256       180
   macro avg      0.277     0.262     0.224       180
weighted avg      0.280     0.256     0.220       180
        """

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        model.summary()

        # -----------------------------
        # Train Model
        # -----------------------------
        model.fit(
            x_train_lstm,
            y_train_cat,
            epochs=100,
            batch_size=32,
            shuffle=False,
            verbose=2
        )

        # -----------------------------
        # Predict on Test Set
        # -----------------------------
        y_pred_prob = model.predict(x_test_lstm)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        print(y_pred_prob)
        print(y_pred_class)
        print(y_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred_class)
        print(f"Accuracy: {acc:.3f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_class, digits=3))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_class)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Quantile Classifier)")
        plt.show()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-n_forecast - 30:-n_forecast], y_train[-30:], label='Training Actual')
        plt.plot(df.index[-n_forecast:], y_test, label='Testing Actual')
        plt.plot(df.index[-n_forecast:], y_pred_class, label='Predicted')
        plt.title(f'{n_forecast}-Day Ahead Stock Return Quantile Prediction')
        plt.xlabel('Date')
        plt.ylabel('Quantiles')
        plt.legend()
        plt.grid(True)
        plt.show()


    def run_XGBoost(self):
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
        """
        Accuracy: 0.228

Classification Report:
              precision    recall  f1-score   support

         0.0      0.258     0.235     0.246        34
         1.0      0.206     0.184     0.194        38
         2.0      0.283     0.371     0.321        35
         3.0      0.147     0.128     0.137        39
         4.0      0.229     0.235     0.232        34

    accuracy                          0.228       180
   macro avg      0.224     0.231     0.226       180
weighted avg      0.222     0.228     0.223       180
        """
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Predict on test set
        y_pred = model.predict(x_test)
        print(y_pred)

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