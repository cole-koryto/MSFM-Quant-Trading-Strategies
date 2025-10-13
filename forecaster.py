import datetime
import json
import os
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
    def __init__(self, symbol):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.__NUM_CLASSES = 5
        self.symbol = symbol

        print("Loading data...")
        self.symbol_data_df = pd.read_parquet(f"./data/{symbol}.parquet")


    def run_LSTM(self, lags):
        print(f"Running LSTM prediction for {self.symbol}")

        # Gathers data
        self.generate_data(lags)

        # Scale features
        num_features = self.x_train.shape[1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(self.x_train)

        # Reshape for LSTM: (samples, timesteps=num_features, features=1)
        x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], num_features, 1))

        # One-hot encode targets
        y_train_cat = to_categorical(self.y_train, num_classes=self.__NUM_CLASSES)

        # Gets LSTM model
        model = self.build_LSTM_model()

        # Fits model
        model.fit(
            x_train_lstm,
            y_train_cat,
            epochs=100,
            batch_size=32,
            shuffle=False,
            verbose=2
        )

        # Predict next day return quantile
        x_last = x_train_lstm[-1].reshape((1, num_features, 1))
        y_pred_prob = model.predict(x_last)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        print(f"Quantile Probability Predictions\n{y_pred_prob}")
        print(f"Quantile Class Predictions\n{y_pred_class}")

        # Create output DataFrame
        prob_cols = [f"Prob_Class_{i}" for i in range(self.__NUM_CLASSES)]
        df_out = pd.DataFrame(y_pred_prob, columns=prob_cols)
        df_out.insert(0, "Predicted_Class", y_pred_class)
        df_out.insert(0, "Symbol", self.symbol)
        return df_out


    def test_LSTM(self, lags, test_size):
        # Gathers test data
        self.generate_data(lags, test_size)
        num_features = self.x_train.shape[1]

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(self.x_train)
        x_test_scaled = scaler.transform(self.x_test)

        # Reshape for LSTM: (samples, timesteps=num_features, features=1)
        x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], num_features, 1))
        x_test_lstm = x_test_scaled.reshape((x_test_scaled.shape[0], num_features, 1))

        # One-hot encode targets
        y_train_cat = to_categorical(self.y_train, num_classes=self.__NUM_CLASSES)

        # Gets LSTM model
        model = self.build_LSTM_model()
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
        self.output_test_metrics(y_pred_class, "LSTM")


    def build_LSTM_model(self):
        # Build LSTM Model
        num_features = self.x_train.shape[1]
        model = Sequential()
        model.add(Input(shape=(num_features, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(self.__NUM_CLASSES, activation='softmax'))

        # model = Sequential()
        # model.add(Input(shape=(lags, 1)))
        # model.add(LSTM(50, return_sequences=True))
        # model.add(Dropout(0.2))  # 20% dropout
        # model.add(LSTM(25))
        # model.add(Dropout(0.2))
        # model.add(Dense(n_classes, activation='softmax'))

        # Train Model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model


    def run_XGBoost(self):
        # Load best params
        with open("testing_results/xgb_best_params.json", "r") as f:
            best_params = json.load(f)

        # Build model with best params
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            **best_params
        )

        # Train on full data
        model.fit(self.x_train, self.y_train)

        # Predict next day
        y_pred = model.predict(self.x_train[-1])
        print(f"Quantile Class Predictions\n{y_pred}")

        # Create output DataFrame
        prob_cols = [f"Prob_Class_{i}" for i in range(self.__NUM_CLASSES)]
        df_out = pd.DataFrame([None for _ in range(self.__NUM_CLASSES)], columns=prob_cols)
        df_out.insert(0, "Predicted_Class", y_pred)
        df_out.insert(0, "Symbol", self.symbol)
        return df_out


    def test_XGBoost(self, lags, test_size):
        # Gathers test data
        self.generate_data(lags, test_size)

        # Gets grid_search model
        grid_search = self.build_XGBoost()

        # Trains model
        grid_search.fit(self.x_train, self.y_train)
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Save to JSON file
        with open("xgb_best_params.json", "w") as f:
            json.dump(grid_search.best_params_, f, indent=4)

        # Predict on test set
        y_pred = model.predict(self.x_test)
        self.output_test_metrics(y_pred, "XGBoost")


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


    def generate_data(self, lags, test_size=None):
        # Creates modeling df
        df = self.symbol_data_df.copy()

        # Creates return quantiles
        df['quantile'] = pd.qcut(df['return'], q=5, labels=False)

        # Create lag features
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['quantile'].shift(i)

        # One-day ahead target
        df['target_1d'] = df['quantile'].shift(-1)

        # Add 2-day, 3-day, and 5-day moving averages for return quintiles
        df['MA_2_q'] = df['quantile'].rolling(window=2).mean()
        df['MA_3_q'] = df['quantile'].rolling(window=3).mean()
        df['MA_5_q'] = df['quantile'].rolling(window=5).mean()

        # Drop NaNs created by shifting
        df.dropna(inplace=True)

        # Features and target
        features = [f'lag_{i}' for i in range(1, lags + 1)]
        features.extend(['MA_2_q', 'MA_3_q', 'MA_5_q'])
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


    def output_test_metrics(self, y_pred, model_name):
        # Accuracy
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")

        # Detailed classification report
        print("\nClassification Report:")
        report = classification_report(self.y_test, y_pred, digits=3)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
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
        filename = os.path.join(results_folder, f"{model_name}_test_results_{timestamp}.txt")
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


    # def save_prediction_LSTM(self, symbol, y_pred_class, y_pred_prob):
    #     # Create folder if it doesn't exist
    #     os.makedirs("testing_results/predictions", exist_ok=True)
    #
    #     # Create filename with symbol + date
    #     today = datetime.date.today().strftime("%Y-%m-%d")
    #     filename = f"testing_results/predictions/{symbol}_{today}.csv"
    #
    #     # Build a DataFrame for readability
    #     df_pred = pd.DataFrame({
    #         "Predicted_Class": y_pred_class,
    #     })
    #
    #     # Add predicted probabilities for each class
    #     for i in range(y_pred_prob.shape[1]):
    #         df_pred[f"Prob_Class_{i}"] = y_pred_prob[:, i]
    #
    #     # Save to CSV
    #     df_pred.to_csv(filename, index=False)
    #     print(f"✅ Saved LSTM predictions to {filename}")
    #
    #
    # def save_xgb_prediction(self, symbol, y_pred_class):
    #     # Create folder if it doesn't exist
    #     os.makedirs("testing_results/predictions", exist_ok=True)
    #
    #     # Create filename using symbol + date
    #     today = datetime.date.today().strftime("%Y-%m-%d")
    #     filename = f"testing_results/predictions/{symbol}_{today}.csv"
    #
    #     # Save only the class predictions
    #     df_pred = pd.DataFrame({"Predicted_Class": y_pred_class})
    #     df_pred.to_csv(filename, index=False)
    #
    #     print(f"✅ Saved XGBoost predictions to {filename}")

