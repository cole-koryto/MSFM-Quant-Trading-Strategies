import pandas as pd
import numpy as np
from scipy import stats
from predictor import Predictor
from dollar_neutral_strategy import Dollar_Neutral

class Backtester():

    def __init__(self, ticker_list, model_type, strategy, initial_value):
        self.ticker_list = ticker_list
        self.model_type = model_type
        self.strategy = strategy # dollar-neutral
        self.initial_value = initial_value
    
    def combine_ticker_dfs(self):
        
        df_preds = pd.DataFrame()
        ticker_df_list = []

        for ticker in self.ticker_list:
            predictor = Predictor(model_type=self.model_type, ticker=ticker)
            df_pred_ticker = predictor.generate_predictions()
            df_pred_ticker.insert(0, "ticker", ticker)
            ticker_df_list.append(df_pred_ticker)
            
        df_preds = pd.concat(ticker_df_list, join='inner')

        return df_preds
    
    def generate_backtest_data(self, data):
        if self.strategy == "dollar-neutral":
            strategy = Dollar_Neutral(data)
            backtest_data = strategy.calculate_weights()
            backtest_data = strategy.calculate_initial_position_size(self.initial_value)

            print(backtest_data.head(10))
            print(backtest_data["position"].value_counts())
            print(backtest_data["position_value"].head(10))

        else:
            backtest_data = data  # fallback if no strategy

        return backtest_data
    
    def calculate_portfolio_value(self, data):

        portfolio_value = pd.Series(self.initial_value, index=data.index.unique())

        dates = data.index.unique()
        for i in range(1, len(dates)):
            idx = dates[i]
            prev_date = dates[i - 1]

            curr_slice = data.loc[idx].copy()
            prev_slice = data.loc[prev_date].copy()

            for ticker in curr_slice["ticker"]:
                # Safely get previous position_value
                prev_pos = prev_slice.loc[prev_slice["ticker"] == ticker, "position_value"]
                prev_pos_value = prev_pos.iloc[0] if not prev_pos.empty else 0

                # Safely get current return
                curr_ret = curr_slice.loc[curr_slice["ticker"] == ticker, "return"]
                curr_ret_value = curr_ret.iloc[0] if not curr_ret.empty else 0

                # Update position_value
                data.loc[(data.index == idx) & (data["ticker"] == ticker), "position_value"] = prev_pos_value * (1 + curr_ret_value)

            # Calculate longs and shorts
            longs_value = data.loc[(data.index == idx) & (data["position"] == 1), "position_value"].sum()
            shorts_value = data.loc[(data.index == idx) & (data["position"] == -1), "position_value"].sum()

            # Assign portfolio value
            portfolio_value.loc[idx] = longs_value + abs(shorts_value)

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

        data = self.combine_ticker_dfs()
        backtest_data = self.generate_backtest_data(data)
        portfolio_value = self.calculate_portfolio_value(backtest_data)
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_value)
        
        return {
            "portfolio_value": portfolio_value,
            "backtest_data": backtest_data,
            "portfolio_metrics": portfolio_metrics
        }