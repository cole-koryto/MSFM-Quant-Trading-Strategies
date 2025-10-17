from forecaster import Forecaster


def main():
    # Makes a prediction of next day of each symbol
    symbols = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]
    for symbol in symbols:
        forecaster = Forecaster(ticker=symbol)
        forecaster.test_XGBoost(lookback=30, test_size=180)      # Can test either model here


if __name__ == "__main__":
    main()