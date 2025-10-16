import pandas as pd
import numpy as np
from scipy import stats
from predictor import Predictor
from dollar_neutral_strategy import Dollar_Neutral

class Backtester():

    def __init__(self, ticker_list, strategy, initial_value):
        self.ticker_list = ticker_list
        self.strategy = strategy # dollar-neutral
        self.initial_value = initial_value
    
    def combine_ticker_dfs(self):
        
        df_preds = pd.DataFrame()
        ticker_df_list = []

        for ticker in self.ticker_list:
            predictor = Predictor(model_type="lstm", ticker=ticker)
            df_pred_ticker = predictor.generate_predictions()
            df_pred_ticker.insert(0, "ticker", self.ticker)
            ticker_df_list.append(df_pred_ticker)
            
        df_preds = pd.concat(ticker_df_list, join='inner')

        return df_preds
    
    def generate_backtest_data(self, data):

        if self.strategy == "dollar-neutral":
            strategy = Dollar_Neutral(data)
            backtest_data = strategy.calculate_weights()
            backtest_data = strategy.calculate_initial_position_size(self.initial_value)
        else:
            None

        return backtest_data
    
    def calculate_portfolio_value(self, data):

        portfolio_value = pd.Series(self.initial_value, index=data.index.unique())

        for idx in data.index.unique()[1:]:
            prev_date = data.index.unique()[data.index.unique().get_loc(idx) - 1]

            curr_slice = data.loc[idx].copy()
            prev_slice = data.loc[prev_date].copy()

            for ticker in curr_slice["ticker"]:
                data.loc[(data.index == idx) & (data["ticker"] == ticker), "position_value"] = (
                    prev_slice[prev_slice["ticker"] == ticker]["position_value"].iloc[0] * 
                    (1 + curr_slice[curr_slice["ticker"] == ticker]["return"].iloc[0])
                )

            # Calculate portfolio value after all dates are processed
            longs_value = data.loc[(data.index == idx) & (data["position"] == 1), "position_value"].sum()
            shorts_value = data.loc[(data.index == idx) & (data["position"] == -1), "position_value"].sum().abs()

            portfolio_value[idx] = longs_value + shorts_value

        return portfolio_value
    
    def calculate_portfolio_metrics(self, data):
        
        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Mean return
        mean_return = returns.mean() * 252

        # Volatility
        volatility = returns.std() * np.sqrt(252)
    
        # Sharpe ratio
        sharpe = (mean_return) / volatility if volatility != 0 else 0

        # Skewness
        skewness = stats.skew(returns)

        # Max drawdown
        cummax = data.cummax()
        drawdown = (data - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - 5% quantile
        var_05 = returns.quantile(0.05)

        portfolio_metrics = {
            "Mean Return": mean_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Skewness": skewness,
            "Max Drawdown": max_drawdown,
            "VaR (5%)": var_05
        }
    
        return portfolio_metrics
    
    def backtest(self):
        # Combine all ticker data
        data = self.combine_ticker_dfs()
        
        # Generate backtest data with weights and initial positions
        backtest_data = self.generate_backtest_data(data)
        
        # Calculate portfolio value over time
        portfolio_value = self.calculate_portfolio_value(backtest_data)
        
        # Calculate performance metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_value)
        
        return {
            "portfolio_value": portfolio_value,
            "backtest_data": backtest_data,
            "portfolio_metrics": portfolio_metrics
        }