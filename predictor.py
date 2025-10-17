import pandas as pd
import numpy as np
import json
from forecaster import Forecaster
from model_loader import ModelLoader
from sklearn.preprocessing import MinMaxScaler

class Predictor():
    
    def __init__(self, model_type, ticker):
        self.model_type = model_type # lstm
        self.ticker = ticker

        # Load time series data for ticker
        self.price_data = pd.read_parquet(f"./data/{ticker}.parquet")

        # Load prediction model for ticker
        self.model_loader = ModelLoader(model_type, ticker)
        self.model = self.model_loader.load_model()

        # self.model.summary()

    def __create_sequences(self, data, lookback):
        X = []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
        return np.array(X)


    def prepare_data(self):
        
        if self.model_type == "lstm":

            # Pull lookback and num_features from .h5
            input_shape = self.model.input_shape
            lookback = input_shape[1]
            num_features = input_shape[2]

            forecaster = Forecaster(self.ticker)
            forecaster.generate_data(lookback=lookback)
            data = forecaster.x_train # no test-size provided

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)

            # Reshape same as training - NO __create_sequences
            data_shaped = data_scaled.reshape((data_scaled.shape[0], num_features, 1))  # (n_samples, 36, 1)
            print(f"data_shaped shape: {data_shaped.shape}")
            
            data_seq = self.__create_sequences(data=data_shaped, lookback=lookback)
            print(f"data_seq shape: {data_seq.shape}")

            return data_seq        

        if self.model_type == "xgb":

            # Load metadata
            with open(f"models/xgb_metadata_{self.ticker}.json", "r") as f:
                meta = json.load(f)
            lookback = meta["lookback"]

            forecaster = Forecaster(self.ticker)
            forecaster.generate_data(lookback=lookback)
            data = forecaster.x_train # no test-size provided

            return data

    def generate_predictions(self):
        model = self.model

        X = self.prepare_data()
        df_pred = X.copy()

        if self.model_type == "lstm":
            y_pred_prob = model.predict(X)
        elif self.model_type == "xgb":
            y_pred_prob = model.predict_proba(X)

        if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
            for i in range(y_pred_prob.shape[1]):
                df_pred[f"prob_class_{i}"] = y_pred_prob[:, i]
            df_pred["predicted_class"] = np.argmax(y_pred_prob, axis=1)

        return df_pred
