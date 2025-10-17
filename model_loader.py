import tensorflow as tf

class ModelLoader():

    def __init__(self, model_type, ticker):
        self.model_type = model_type # lstm
        self.ticker = ticker

    def load_model(self):
        model_path = f"models/lstm_model_{self.ticker}"
        model = tf.keras.models.load_model(model_path)
        return model