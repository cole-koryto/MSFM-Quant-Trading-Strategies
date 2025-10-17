import json
import tensorflow as tf
from xgboost import XGBClassifier

class ModelLoader():

    def __init__(self, model_type, ticker):
        self.model_type = model_type # lstm or xgb
        self.ticker = ticker

    def load_model(self):

        if self.model_type == "lstm":
            model_path = f"models/lstm_model_{self.ticker}"
            model = tf.keras.models.load_model(model_path)
            return model
    
        elif self.model_type == "xgb":
            model = XGBClassifier()
            model.load_model(f"models/xgb_model_{self.ticker}.json")
            return model