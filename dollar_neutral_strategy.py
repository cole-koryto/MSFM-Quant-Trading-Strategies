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
        backtest_df["weight"] = 0
        for date in backtest_df.index.unique():
            
            date_slice = backtest_df.loc[date]

            longs = date_slice["position"] == 1
            shorts = date_slice["position"] == -1

            if longs.any():
                long_tickers = date_slice[longs]["ticker"]
                backtest_df.loc[(backtest_df.index == date) & (backtest_df["ticker"].isin(long_tickers)), "weight"] = (
                    date_slice.loc[longs, "prob_class_4"] /
                    date_slice.loc[longs, "prob_class_4"].sum()
                ) * date_slice.loc[longs, "position"]

            if shorts.any():
                short_tickers = date_slice[shorts]["ticker"]
                backtest_df.loc[(backtest_df.index == date) & (backtest_df["ticker"].isin(short_tickers)), "weight"] = (
                    date_slice.loc[shorts, "prob_class_4"] /
                    date_slice.loc[shorts, "prob_class_4"].sum()
                ) * date_slice.loc[shorts, "position"]

        return backtest_df