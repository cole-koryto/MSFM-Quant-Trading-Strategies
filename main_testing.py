from forecaster import Forecaster
from get_nsdq_tickers import get_nasdaq_100

def main():
    # Makes a prediction of next day of each symbol
    # tickers = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]
    tickers = get_nasdaq_100()
    for ticker in tickers:
        forecaster = Forecaster(ticker=ticker)
        # forecaster.test_LSTM(lookback=30, test_size=180)      # Can also test XGBoost here
        forecaster.test_XGBoost(lookback=30, test_size=180)

if __name__ == "__main__":
    main()