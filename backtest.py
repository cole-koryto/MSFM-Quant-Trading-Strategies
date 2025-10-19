import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from backtester import Backtester
from price_loader import PriceLoader

ticker_list = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]

loader = PriceLoader(ticker_list)
loader.save_to_parquet((datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))

backtester = Backtester(ticker_list, "xgb", "dollar-neutral", 1_000_000)

result = backtester.backtest()
metrics_summary = {k: v for k, v in result["portfolio_metrics"].items() if not hasattr(v, "__len__") or isinstance(v, str)}
metrics_df = pd.DataFrame(metrics_summary, index=[0]).T
print(metrics_df)

portfolio_value = result["portfolio_value"]

plt.figure(figsize=(12,6))
plt.plot(portfolio_value.index, portfolio_value.values, marker='o')
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()

drawdown = result["portfolio_metrics"]["Drawdown (Time Series)"]
drawdown_stats = result["portfolio_metrics"]["Drawdown (Stats)"]

# Create the drawdown time series plot
plt.figure(figsize=(10, 5))
plt.plot(drawdown, label='Drawdown', linewidth=2)
plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)

# Add title and labels
plt.title("Portfolio Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print drawdown diagnostics
print(drawdown_stats)