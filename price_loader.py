"""
price_loader.py

Responsible for using yfinance to to download 
daily adjusted close prices for all S&P 500 tickers 
(use the list of S&P 500 tickers from today, do not go back point-in-time. 
i.e some of the tickers you have today won't be 
the same as the real ones from 2015 - it doesn't matter)

Data will be:
1. stored locally using parquets
2. implemented using PriceLoader class to manage access
3. API limits with batching
4. Tickers will be dropped with sparse or missing data

Libraries used: pandas (parquetting, dataframe), yfinance ()
"""
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time
import os

class PriceLoader:

    def __init__(self, tickers):
        self.tickers = tickers


    @staticmethod
    def yield_batches(tickers, batch_size=25):
        """
        Generator function that yields tickers
        in batch sizes of 25 (typically) for 
        batching. Will be used for yfinance.
        """
        # yield yfinance data by batches
        for i in range(0, len(tickers), batch_size):
            yield tickers[i: min(i + batch_size, len(tickers))]
    
    
    def save_to_parquet(self, start, end):
        """
        Uses ticker values from 
        get_tickers() and then batches
        the tickers from yfinance. Then
        saves it from parquet.
        """
        # log start time
        start_time = time.time()

        # create directory
        if not os.path.exists("data/"):
            os.makedirs("data")
        
        # create list of concatenated S&P dataframes
        snp_dfs = []

        # batch number for logging
        batch_num = 0
        
        for batch in self.yield_batches(self.tickers, 25):
            print(f"Downloading batch {batch_num}")
            
            try:
                # get only the closing prices (This is really adjusted close)
                df = yf.download(tickers=batch, start=start, end=end, auto_adjust=True)["Open"]

                # add dataframe to the to-be concatenated dataframe
                snp_dfs.append(df)

                # logging
                print(f"batch {batch_num} complete!")
            except Exception:
                print(f"issue with downloading one of the following tickers: {self.tickers}")

            batch_num += 1
        print(f"total_batches: {batch_num}")
        snp_dfs = pd.concat(snp_dfs, axis=1)

        # send dataframe to respective parquets
        for ticker in self.tickers:

            # ensure that there is in fact data for ticker
            if ticker in snp_dfs.columns:

                # separate df into its own data file as parquets
                ticker_df = snp_dfs.loc[:, [ticker]]
                ticker_df = ticker_df.rename(columns={f"{ticker}": "price"})
                ticker_df.columns.name = None

                # create return column
                ticker_df["return"] = ticker_df["price"].pct_change()

                # use ticker for the name
                ticker_df.to_parquet(f"data/{ticker}.parquet")
                print(f"{ticker} parquet created!")

        # log end time
        end = time.time()

        # see how long batching took
        print(f"total batching and data loading time took {end - start_time:0.2f} seconds")


if __name__ == "__main__":
    loader = PriceLoader(['6EZ25.CME', '6JZ25.CME', 'ZWZ25.CBT'])
    loader.save_to_parquet((datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))