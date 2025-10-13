import datetime
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
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        print("Loading data...")
        self.price_df = pd.read_parquet("./data/ES=F.parquet")
        self.price_df["return"] = self.price_df["Price"].pct_change()
        self.__NUM_CLASSES = 5
        self.test_data_generated = False


    def test_LSTM(self, lags, test_size):
        # Gathers test data if it is not already present
        if not self.test_data_generated:
            self.generate_test_data(lags, test_size)
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

        # Build LSTM Model
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


    def test_XGBoost(self, lags, test_size):
        # Gathers test data if it is not already present
        if not self.test_data_generated:
            self.generate_test_data(lags, test_size)

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
        grid_search.fit(self.x_train, self.y_train)
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Predict on test set
        y_pred = model.predict(self.x_test)
        self.output_test_metrics(y_pred, "XGBoost")


    def generate_test_data(self, lags, test_size):
        # Creates modeling df
        df = self.price_df.copy()

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

        # Train/test split
        self.x_train = x[:-test_size]
        self.x_test = x[-test_size:]
        self.y_train = y[:-test_size]
        self.y_test = y[-test_size:]
        self.test_data_generated = True


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
        filename = os.path.join(results_folder, f"test_results_{timestamp}.txt")
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