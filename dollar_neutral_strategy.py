import pandas as pd
from strategy import Strategy

class Dollar_Neutral(Strategy):
        
    def __init__(self, data):
        self.data = data
    
    def calculate_weights(self):
        
        backtest_df = self.data

        backtest_df["position"] = 0
        backtest_df.loc[backtest_df["predicted_class"] == 4, "position"] = 1 # longs
        backtest_df.loc[backtest_df["predicted_class"] == 0, "position"] = -1 # shorts

        # Assign overweights and underweights
        backtest_df["weight"] = 0.0

        # Group by date to assign weights
        for date, group in backtest_df.groupby(backtest_df.index):
            longs = group["position"] == 1
            shorts = group["position"] == -1

            # Assign long weights
            if longs.any():
                long_probs = group.loc[longs, "prob_class_4"]
                long_weights = (long_probs / long_probs.sum()) * 1  # longs = +1
                backtest_df.loc[long_weights.index, "weight"] = long_weights

            # Assign short weights
            if shorts.any():
                short_probs = group.loc[shorts, "prob_class_4"]
                short_weights = (short_probs / short_probs.sum()) * -1  # shorts = -1
                backtest_df.loc[short_weights.index, "weight"] = short_weights

            return backtest_df
    
    def calculate_initial_position_size(self, initial_value):

        backtest_df = self.data.copy()

        initial_longs_value = initial_value / 2
        initial_shorts_value = initial_value / 2

        first_date = backtest_df.index.unique()[0]
        first_date_mask = backtest_df.index == first_date
        
        longs_mask = first_date_mask & (backtest_df["position"] == 1)
        shorts_mask = first_date_mask & (backtest_df["position"] == -1)
        
        backtest_df.loc[longs_mask, "position_value"] = initial_longs_value * backtest_df.loc[longs_mask, "weight"]
        backtest_df.loc[shorts_mask, "position_value"] = initial_shorts_value * backtest_df.loc[shorts_mask, "weight"]
    
        return backtest_df