from forecaster import Forecaster


def main():
    # Makes a prediction of next day of each symbol
    symbols = ["ES=F", "^SPX"]
    results = []
    for symbol in symbols:
        forecaster = Forecaster(symbol=symbol)
        df_pred = forecaster.test_LSTM(lags=30, test_size=180)      # Can also test XGBoost here
        results.append(df_pred)


if __name__ == "__main__":
    main()