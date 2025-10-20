from datetime import datetime, timedelta
from price_loader import PriceLoader
from forecaster import Forecaster
from panel_forecaster import PanelForecaster
from get_nsdq_tickers import get_nasdaq_100

def main():
    # Makes a prediction of next day of each symbol
    # tickers = ["6EZ25.CME", "6JZ25.CME", "ZWZ25.CBT"]
    tickers = get_nasdaq_100()

    loader = PriceLoader(tickers)
    loader.save_to_parquet((datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))

    # for ticker in tickers:
    #     forecaster = Forecaster(ticker=ticker)
    #     # forecaster.test_LSTM(lookback=30, test_size=180)      # Can also test XGBoost here
    #     forecaster.test_XGBoost(lookback=30, test_size=180)

    panel_forecaster = PanelForecaster(tickers, "nasdaq100")
    panel_forecaster.test_LightGBM(lags=10, test_share=0.2)

if __name__ == "__main__":
    main()