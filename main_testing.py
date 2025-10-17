from forecaster import Forecaster


def main():
    # Makes a prediction of next day of each symbol
    tickers = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]
    for ticker in tickers:
        forecaster = Forecaster(ticker=ticker)
        # forecaster.test_LSTM(lookback=30, test_size=180)      # Can also test XGBoost here
        forecaster.test_XGBoost(lookback=30, test_size=180)

if __name__ == "__main__":
    main()