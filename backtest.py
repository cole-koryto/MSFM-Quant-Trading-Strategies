import pandas as pd
import matplotlib.pyplot as plt
from backtester import Backtester

ticker_list = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]

backtester = Backtester(ticker_list, "xgb", "dollar-neutral", 1_000_000)

result = backtester.backtest()
metrics_df = pd.DataFrame(result["portfolio_metrics"], index=[0]).T
portfolio_value = result["portfolio_value"]
print(metrics_df)

plt.figure(figsize=(12,6))
plt.plot(portfolio_value.index, portfolio_value.values, marker='o')
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()