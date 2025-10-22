import datetime
import json
import os
import pprint

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class PanelForecaster:
    def __init__(self, ticker_list, model_name):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.__NUM_CLASSES = 5
        self.ticker_list = ticker_list
        self.model_name = model_name

        print("Loading data...")

        ticker_df_list = []

        for ticker in self.ticker_list:
            ticker_data_df = pd.read_parquet(f"./data/{ticker}.parquet")
            ticker_data_df.insert(0, "ticker", ticker)
            ticker_df_list.append(ticker_data_df)
            
        print("Merging data...")
        self.data_df = pd.concat(ticker_df_list, join='inner')
        print("Data merged...")

        self.data_df.to_csv(f"./data/{model_name}")

    def build_LightGBM(self, x_train, y_train, x_valid, y_valid):
        """Build and optimize LightGBM using Optuna with train/validation split."""

        # Drop non-numeric columns from passed data (not self.x_train)
        x_train = x_train.drop(['ticker'], axis=1)
        x_valid = x_valid.drop(['ticker'], axis=1)

        def objective(trial):
            """Objective function for Optuna hyperparameter optimization."""
            params = {
                'objective': 'multiclass',
                'num_class': 5,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_threads': 4,
            }

            train_set = lgb.Dataset(x_train, label=y_train)
            valid_set = lgb.Dataset(x_valid, label=y_valid)

            model = lgb.train(
                params,
                train_set,
                valid_sets=[valid_set],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(period=0)
                ]
            )

            preds = model.predict(x_valid)
            preds_labels = preds.argmax(axis=1)
            acc = accuracy_score(y_valid, preds_labels)
            
            return 1.0 - acc  # minimize error

        # Create study and optimize
        sampler = TPESampler(seed=25)
        pruner = MedianPruner()

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(objective, n_trials=50, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        print(f"Best Accuracy: {1 - study.best_value:.4f}")
        print(f"Best Parameters: {best_params}")

        # Save best parameters
        os.makedirs("models", exist_ok=True)
        with open(f"models/lgb_best_params_{self.model_name}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        # Train final model with best parameters on full training data
        train_set = lgb.Dataset(x_train, label=y_train)
        
        final_params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_threads': 4,
            **best_params
        }

        final_model = lgb.train(
            final_params,
            train_set,
            num_boost_round=1000,
            callbacks=[
                lgb.log_evaluation(period=0)
            ]
        )

        return final_model, best_params


    def test_LightGBM(self, lags, test_share):
        """Build, optimize, and test LightGBM model."""

        # Generate train/test data
        self.generate_data(lags, test_share)

        # Create validation split from training data
        x_train, x_valid, y_train, y_valid = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=25, stratify=self.y_train
        )

        # Build and optimize model with Optuna
        model, best_params = self.build_LightGBM(x_train, y_train, x_valid, y_valid)

        # Prepare test data
        x_test = self.x_test.drop(['ticker'], axis=1)

        # Predict on test set
        preds = model.predict(x_test)
        y_pred = preds.argmax(axis=1)

        # Output metrics
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        self.LightGBM_output_test_metrics(y_pred, "LightGBM")


    def run_LightGBM(self, lags):
        """Load best params and train final production model."""
        # Generate data
        self.generate_data(lags)

        # Load best parameters from previous optimization
        with open(f"models/lgb_best_params_{self.model_name}.json", "r") as f:
            best_params = json.load(f)

        # Prepare data (drop non-numeric columns)
        x_train = self.x_train.drop(['ticker'], axis=1)

        # Build and train model with best parameters
        train_set = lgb.Dataset(x_train, label=self.y_train)

        final_params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_threads': 4,
            **best_params
        }

        model = lgb.train(
            final_params,
            train_set,
            num_boost_round=1000,
            callbacks=[
                lgb.log_evaluation(period=0)
            ]
        )

        # Save trained model
        os.makedirs("models", exist_ok=True)
        model.save_model(f"models/lgb_model_{self.model_name}.txt")

        # Save metadata
        metadata = {
            "lags": lags,
            "features": list(x_train.columns),
            "model_type": "LGBMBooster",
            "best_params": best_params
        }

        with open(f"models/lgb_metadata_{self.model_name}.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model trained and saved: models/lgb_model_{self.model_name}.txt")
        return model


    def generate_data(self, lags, test_share=None):
        # Creates modeling df
        df = self.data_df.copy()

        def expanding_quantiles(x):
            q = self.__NUM_CLASSES
            result = pd.Series(index=x.index, dtype=int)
            for i in range(len(x)):
                # Only take values up to current point
                subset = x.iloc[:i+1]
                result.iloc[i] = pd.qcut(subset, q=q, labels=False, duplicates='drop')[-1]
            return result
            
        # Creates return quantiles within each ticker
        df['quantile'] = df.groupby('ticker')['return'].transform(lambda x: expanding_quantiles(x))

        # Create lag features
        for i in range(1, lags + 1):
            # Lag for quantiles within each ticker
            df[f'quantile_lag_{i}'] = df.groupby('ticker')['quantile'].shift(i)
            
            # Lag for returns within each ticker
            df[f'return_lag_{i}'] = df.groupby('ticker')['return'].shift(i)

        # One-day ahead target
        df['target_1d'] = df.groupby('ticker')['quantile'].shift(-1)

        # Add 2-day, 3-day, and 5-day moving averages for returns
        windows = [2, 3, 5, 10, 21]

        for window in windows:
            df[f'MA_{window}_ret'] = df.groupby('ticker')['return'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'MA_{window}_q'] = df.groupby('ticker')['quantile'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'mom_{window}d'] = df.groupby('ticker')['return'].transform(
                lambda x: x.rolling(window=window).sum()
            )
            df[f'vol_{window}d'] = df.groupby('ticker')['return'].transform(
                lambda x: x.rolling(window=window).std()
            )
            
        # Daily z-score of return and volatility across tickers
        df['cross_z_ret'] = df.groupby(level=0)['return'].transform(lambda x: (x - x.mean()) / x.std())
        df['cross_z_vol'] = df.groupby(level=0)['vol_21d'].transform(lambda x: (x - x.mean()) / x.std())    

        # Drop NaNs created by shifting
        df.dropna(inplace=True)

        y = df['target_1d']
        X = df.drop('target_1d', axis=1)
        
        def time_based_split(X, y, test_share=0.2):
            """
            Split X and y into train/test sets by ticker and time.
            The last `test_share` fraction of each ticker’s data goes to test.
            """
            x_train_list, x_test_list = [], []
            y_train_list, y_test_list = [], []

            for ticker, group in X.groupby('ticker'):
                print(f"Processing {ticker}...") 
                idx = group.index
                n = len(idx)
                if n < 2:
                    # Too few observations, skip test split
                    train_idx = idx
                    test_idx = []
                else:
                    split_point = int(n * (1 - test_share))
                    train_idx = idx[:split_point]
                    test_idx = idx[split_point:]

                x_train_list.append(X.loc[train_idx])
                y_train_list.append(y.loc[train_idx])
                if len(test_idx) > 0:
                    x_test_list.append(X.loc[test_idx])
                    y_test_list.append(y.loc[test_idx])

            # Combine all tickers’ splits
            x_train = pd.concat(x_train_list)
            y_train = pd.concat(y_train_list)
            x_test = pd.concat(x_test_list) if x_test_list else None
            y_test = pd.concat(y_test_list) if y_test_list else None

            return x_train, x_test, y_train, y_test

        # Train/test split if needed
        if test_share:
            self.x_train, self.x_test, self.y_train, self.y_test = time_based_split(X, y, test_share)
        else:
            self.x_train, self.y_train = X, y
            self.x_test = self.y_test = None


    def output_test_metrics(self, y_pred, model_name):
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