import datetime
import json
import os
import pprint

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Forecaster:
    def __init__(self, ticker):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.__NUM_CLASSES = 5
        self.ticker = ticker

        print("Loading data...")
        self.ticker_data_df = pd.read_parquet(f"./data/{ticker}.parquet")


    def run_LSTM(self, lookback):
        print(f"Running LSTM prediction for {self.ticker}")

        # Gathers data
        self.generate_data(lookback)

        # Scale features
        num_features = self.x_train.shape[1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(self.x_train)

        # Create rolling windows manually for LSTM
        n_samples = x_train_scaled.shape[0] - lookback + 1
        x_train_lstm = np.zeros((n_samples, lookback, num_features))
        for i in range(n_samples):
            x_train_lstm[i] = x_train_scaled[i : i + lookback]

        # One-hot encode targets
        y_train_cat = to_categorical(self.y_train[lookback-1:], num_classes=self.__NUM_CLASSES)

        # Gets LSTM model
        model = self.build_LSTM_model(lookback)

        # Fits model
        model.fit(
            x_train_lstm,
            y_train_cat,
            epochs=50,
            batch_size=32,
            shuffle=False,
            verbose=2
        )
        os.makedirs("models", exist_ok=True)
        model.save(f"models/lstm_model_{self.ticker}", save_format='tf')

        # Predict next day return quantile
        x_last = x_train_lstm[-1:].reshape(1, lookback, num_features)
        y_pred_prob = model.predict(x_last)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        print(f"Quantile Probability Predictions\n{y_pred_prob}")
        print(f"Quantile Class Predictions\n{y_pred_class}")

        # Create output DataFrame
        prob_cols = [f"Prob_Class_{i}" for i in range(self.__NUM_CLASSES)]
        df_out = pd.DataFrame(y_pred_prob, columns=prob_cols)
        df_out.insert(0, "Predicted_Class", y_pred_class)
        df_out.insert(0, "Ticker", self.ticker)
        return df_out


    def test_LSTM(self, lookback, test_size):
        # Generate train/test data
        self.generate_data(lookback, test_size)
        num_features = self.x_train.shape[1]

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(self.x_train)
        x_test_scaled = scaler.transform(self.x_test)

        # Create rolling windows for LSTM
        n_train_samples = x_train_scaled.shape[0] - lookback + 1
        x_train_lstm = np.zeros((n_train_samples, lookback, num_features))
        for i in range(n_train_samples):
            x_train_lstm[i] = x_train_scaled[i:i+lookback]

        y_train_cat = to_categorical(self.y_train[lookback-1:], num_classes=self.__NUM_CLASSES)

        n_test_samples = x_test_scaled.shape[0] - lookback + 1
        x_test_lstm = np.zeros((n_test_samples, lookback, num_features))
        for i in range(n_test_samples):
            x_test_lstm[i] = x_test_scaled[i:i+lookback]

        y_test_cat = to_categorical(self.y_test[lookback-1:], num_classes=self.__NUM_CLASSES)

        # Build model with proper input shape
        model = self.build_LSTM_model(lookback)

        # Train model
        model.fit(
            x_train_lstm,
            y_train_cat,
            epochs=100,
            batch_size=32,
            shuffle=False,
            verbose=2
        )

        # Predict on Test Set
        y_pred_prob = model.predict(x_test_lstm)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        self.LSTM_test_metrics(y_pred_class, "LSTM",lookback=lookback)


    def build_LSTM_model(self, lookback):
        # Build LSTM Model
        num_features = self.x_train.shape[1]

        # model = Sequential()
        # model.add(Input(shape=(lookback, num_features)))
        # model.add(LSTM(units=50))
        # model.add(Dense(self.__NUM_CLASSES, activation='softmax'))

        model = Sequential()
        model.add(Input(shape=(lookback, num_features)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))  # 20% dropout
        model.add(LSTM(25))
        model.add(Dropout(0.2))
        model.add(Dense(self.__NUM_CLASSES, activation='softmax'))

        # Train Model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model


    def run_XGBoost(self, lookback):
        # Gathers  data
        self.generate_data(lookback)

        # Load best params
        with open(f"models/xgb_best_params_{self.ticker}.json", "r") as f:
            best_params = json.load(f)

        # Build model with best params
        model = XGBClassifier(
            **best_params,  # unpack tuned params
            objective='multi:softprob',
            num_class=5,
            tree_method='hist',
            eval_metric='mlogloss',
            random_state=25,
            n_jobs=-1
        )

        # Train on full data
        model.fit(self.x_train, self.y_train)

        # Save model
        os.makedirs("models", exist_ok=True)
        model.save_model(f"models/xgb_model_{self.ticker}.json")

        metadata = {
            "lookback": lookback,
            "features": list(self.x_train.columns),
            "model_type": "XGBRegressor"
        }

        with open(f"models/xgb_metadata_{self.ticker}.json", "w") as f:
            json.dump(metadata, f)

        # Predict next day
        x_last = self.x_train.iloc[-1].values.reshape(1, -1)
        y_pred_prob = model.predict_proba(x_last)
        y_pred_class = model.predict(x_last)
        print(f"Quantile Probability Predictions\n{y_pred_prob}")
        print(f"Quantile Class Predictions\n{y_pred_class}")

        # Create output DataFrame
        prob_cols = [f"Prob_Class_{i}" for i in range(self.__NUM_CLASSES)]
        df_out = pd.DataFrame(y_pred_prob, columns=prob_cols)
        df_out.insert(0, "Predicted_Class", y_pred_class)
        df_out.insert(0, "Ticker", self.ticker)
        return df_out


    def test_XGBoost(self, lookback, test_size):
        # Gathers test data
        self.generate_data(lookback, test_size)

        # Gets grid_search model
        grid_search = self.build_XGBoost()

        # Trains model
        grid_search.fit(self.x_train, self.y_train)
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Save to JSON file
        os.makedirs("models", exist_ok=True)
        with open(f"models/xgb_best_params_{self.ticker}.json", "w") as f:
            json.dump(grid_search.best_params_, f, indent=4)

        # Predict on test set
        y_pred = model.predict(self.x_test)
        self.XGBoost_output_test_metrics(y_pred, "XGBoost")


    def build_XGBoost(self):
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
            objective='multi:softprob',  # provides actual class probabilities
            num_class=5,
            tree_method='hist',  # faster than approx
            eval_metric='mlogloss',  # good for multi-class
            random_state=25,
            n_jobs=-1
        )
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',  # could also do neg_log_loss if we focus on probability calibration
            verbose=2,
            n_jobs=-1)

        return grid_search


    def generate_data(self, lookback, test_size=None):
        # Creates modeling df
        df = self.ticker_data_df.copy()

        # Creates return quantiles
        df['quantile'] = pd.qcut(df['return'], q=3, labels=False)

        # Create lag features
        for i in range(1, lookback + 1):
            df[f'quantile_lag_{i}'] = df['quantile'].shift(i)
            df[f'return_lag_{i}'] = df['return'].shift(i)

        # One-day ahead target
        df['target_1d'] = df['quantile'].shift(-1)

        # Add 2-day, 3-day, and 5-day moving averages for returns
        df['MA_2_ret'] = df['return'].rolling(window=2).mean()
        df['MA_3_ret'] = df['return'].rolling(window=3).mean()
        df['MA_5_ret'] = df['return'].rolling(window=5).mean()

        # Add 2-day, 3-day, and 5-day moving averages for return quintiles
        df['MA_2_q'] = df['quantile'].rolling(window=2).mean()
        df['MA_3_q'] = df['quantile'].rolling(window=3).mean()
        df['MA_5_q'] = df['quantile'].rolling(window=5).mean()

        # Drop NaNs created by shifting
        df.dropna(inplace=True)

        # Features and target
        features = [f'quantile_lag_{i}' for i in range(1, lookback + 1)]
        features.extend([f'return_lag_{i}' for i in range(1, lookback + 1)])
        features.extend(['MA_2_q', 'MA_3_q', 'MA_5_q'])
        features.extend(['MA_2_ret', 'MA_3_ret', 'MA_5_ret'])
        x = df[features]
        y = df['target_1d']

        # Train/test split if needed
        if test_size:
            self.x_train = x[:-test_size]
            self.x_test = x[-test_size:]
            self.y_train = y[:-test_size]
            self.y_test = y[-test_size:]
        else:
            self.x_train = x
            self.y_train = y


    def LSTM_output_test_metrics(self, y_pred, model_name, lookback=1):
        # Align y_test with predictions
        y_test_aligned = self.y_test[lookback-1:]

        # Accuracy
        acc = accuracy_score(y_test_aligned, y_pred)
        print(f"Accuracy: {acc:.3f}")

        # Detailed classification report
        print("\nClassification Report:")
        report = classification_report(y_test_aligned, y_pred, digits=3)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test_aligned, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Quantile Classifier)")
        plt.show()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_train.index, self.y_train, label='Training Actual')
        plt.plot(y_test_aligned.index, y_test_aligned, label='Testing Actual')
        plt.plot(y_test_aligned.index, y_pred, label='Predicted')
        plt.title('Return Quantile Prediction')
        plt.xlabel('Date')
        plt.ylabel('Return Quantile')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create results folder and file
        results_folder = "testing_results"
        os.makedirs(results_folder, exist_ok=True)

        # Save output to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(results_folder, f"{model_name}_test_results_{self.ticker}_{timestamp}.txt")
        with open(filename, "w") as f:
            f.write(f"{model_name} Quantile Classifier Testing Results\n")
            f.write("========================================\n\n")
            f.write(f"Date/Time: {timestamp}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n")
        print(f"✅ Testing results saved to: {filename}")

    def XGBoost_output_test_metrics(self, y_pred, model_name):
        # Accuracy
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")

        # Detailed classification report
        print("\nClassification Report:")
        report = classification_report(self.y_test, y_pred, digits=3)
        print(report)

        # Confusion matrix
        # cm = confusion_matrix(self.y_test, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
        # disp.plot(cmap="Blues")
        # plt.title("Confusion Matrix (Quantile Classifier)")
        # plt.show()

        cm = confusion_matrix(self.y_test, y_pred)
        classes = np.unique(self.y_test)  # [0, 1, 2, 3]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Quantile Classifier)")
        plt.show()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_train.index, self.y_train, label='Training Actual')
        plt.plot(self.y_test.index, self.y_test, label='Testing Actual')
        plt.plot(self.y_test.index, y_pred, label='Predicted')
        plt.title('Return Quantile Prediction')
        plt.xlabel('Date')
        plt.ylabel('Return Quantile')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create results folder and file
        results_folder = "testing_results"
        os.makedirs(results_folder, exist_ok=True)

        # Save output to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(results_folder, f"{model_name}_test_results_{self.ticker}_{timestamp}.txt")
        with open(filename, "w") as f:
            f.write(f"{model_name} Quantile Classifier Testing Results\n")
            f.write("========================================\n\n")
            f.write(f"Date/Time: {timestamp}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n")
        print(f"✅ Testing results saved to: {filename}")