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
    
    def calculate_portfolio_returns(self, data):
        if self.strategy == "dollar-neutral":
            strategy = Dollar_Neutral(data)
            backtest_data = strategy.calculate_weights()

            # Multiply position * weight * return per ticker
            backtest_data["contribution"] = backtest_data["position"] * backtest_data["weight"] * backtest_data["return"]
            
            # Sum contributions across tickers for each date
            portfolio_returns = backtest_data.groupby(backtest_data.index)["contribution"].sum()

            return portfolio_returns
    
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

        # Drawdown and stats (max, bottom date, peak date, recovery date, and duration to recovery)
        def compute_drawdown_stats(returns):
            cum_rets = (1 + returns).cumprod()
            rolling_max = cum_rets.cummax()
            drawdown = (cum_rets - rolling_max) / rolling_max

            # Compute drawdown stats
            max_dd = drawdown.min()
            bottom_date = drawdown.idxmin()
            peak_date = returns.loc[:bottom_date].idxmax()
            recovery_date = drawdown.loc[bottom_date:].gt(-1e-6).idxmax()
            
            if isinstance(recovery_date, (float, np.floating)):  # if not found
                recovery_date = pd.NaT

            duration = recovery_date - peak_date if pd.notna(recovery_date) else pd.NaT
            
            # Create DataFrame for display
            stats = pd.DataFrame([{
                'Max Drawdown': max_dd,
                'Peak': peak_date,
                'Bottom': bottom_date,
                'Recover': recovery_date,
                'Duration (to Recover)': duration
            }])

            # Format it like your example
            formatted = stats.assign(
                **{
                    'Max Drawdown': stats['Max Drawdown'].map('{:.2%}'.format),
                    'Peak': stats['Peak'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''),
                    'Bottom': stats['Bottom'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''),
                    'Recover': stats['Recover'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''),
                    'Duration (to Recover)': stats['Duration (to Recover)'].apply(
                        lambda x: f'{x.days:,} days' if pd.notna(x) else ''
                    )
                }
            ).T

            formatted.columns = ['Value']

            return drawdown, formatted
        
        drawdown, drawdown_stats = compute_drawdown_stats(returns)
        
        # Value at Risk (VaR) - 5% quantile
        var_05 = returns.quantile(0.05)

        portfolio_metrics = {
            "Mean Return": mean_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Skewness": skewness,
            "Drawdown (Time Series)": drawdown,
            "Drawdown (Stats)": drawdown_stats,
            "Max Drawdown": drawdown.min(),
            "VaR (5%)": var_05
        }
    
        return portfolio_metrics
    
    def backtest(self):

        data = self.combine_ticker_dfs()
        portfolio_returns = self.calculate_portfolio_returns(data)
        initial_value = self.initial_value
        portfolio_value = initial_value * (1 + portfolio_returns).cumprod()
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_value)
        
        return {
            "portfolio_value": portfolio_value,
            "portfolio_metrics": portfolio_metrics
        }