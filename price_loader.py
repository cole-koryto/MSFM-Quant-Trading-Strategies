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
import json

CACHE_FILE = "cached_tickers.json"

def get_yahoo_futures_tickers():
    """Scrape Yahoo Finance for all listed futures symbols."""
    try:
        urls = [
            "https://finance.yahoo.com/markets/futures/",
            "https://finance.yahoo.com/markets/commodities/",
            "https://finance.yahoo.com/markets/currencies/"
        ]
        tickers = set()
        for url in urls:
            tables = pd.read_html(url)
            for tbl in tables:
                if "Symbol" in tbl.columns:
                    tickers.update(tbl["Symbol"].dropna().tolist())
        return sorted(tickers)
    except Exception as e:
        print(f"⚠️ Could not fetch futures tickers: {e}")
        return []
    

# Step 0: Ensure baseline cached tickers exist
if not os.path.exists(CACHE_FILE):
    baseline_tickers = [
        # Futures
        "ES=F", "NQ=F", "YM=F", "GC=F", "SI=F", "CL=F",
        # Optionable equities
        "AAPL", "MSFT", "TSLA", "AMZN", "SPY", "QQQ", "NVDA", "META", "GOOG", "NFLX"
    ]
    with open(CACHE_FILE, "w") as f:
        json.dump(baseline_tickers, f)
    print(f"✅ Baseline cached ticker file created: {CACHE_FILE}")

def get_tickers_cached():
    """Load cached tickers from JSON."""
    with open(CACHE_FILE) as f:
        return json.load(f)

def chunk_list(lst, n=50):
    """Split a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_biggest_losers(n=3, lookback_days=5, chunk_size=50, delay=0.2):
    """
    Returns top N biggest losers from the previous trading day.
    Fetches tickers dynamically with fallback to cached list.
    """
    # Step 1: Try to fetch dynamic tickers
    tickers = []
    try:
        tickers = get_dynamic_tickers()
        if tickers:
            with open(CACHE_FILE, "w") as f:
                json.dump(tickers, f)
    except Exception as e:
        print(f"⚠️ Dynamic fetch failed: {e}")

    # Step 2: fallback to cached tickers
    if not tickers:
        print("⚠️ Using cached tickers.")
        tickers = get_tickers_cached()

    print(f"✅ {len(tickers)} tickers will be analyzed.")

    # Step 3: Download prices in chunks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    all_changes = []

    for batch in chunk_list(tickers, chunk_size):
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=True
            )
            for t in batch:
                try:
                    close = data[t]["Close"].dropna()
                    if len(close) >= 2:
                        pct_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                        all_changes.append((t, pct_change))
                except Exception:
                    continue
            time.sleep(delay)  # small pause between batches
        except Exception as e:
            print(f"⚠️ Failed batch {batch[:5]}: {e}")
            time.sleep(delay * 5)

    if not all_changes:
        raise RuntimeError("❌ No price data fetched. Check cached tickers or Yahoo rate limits.")

    # Step 4: Sort by % change and return top N losers
    df = pd.DataFrame(all_changes, columns=["Ticker", "Daily % Change"])
    df = df.sort_values("Daily % Change").reset_index(drop=True)
    losers = df.head(n)

    print("\n📉 Biggest Losers from Previous Day:")
    print(losers)
    return losers


import os
import time
import pandas as pd
import yfinance as yf

class PriceLoader:

    def __init__(self, tickers):
        self.tickers = tickers

    @staticmethod
    def yield_batches(tickers, batch_size=25):
        for i in range(0, len(tickers), batch_size):
            yield tickers[i: min(i + batch_size, len(tickers))]

    def save_to_parquet(self, start, end, retries=2, delay=1):
        """
        Downloads tickers in batches, skips empty or failed downloads,
        and saves each ticker as its own parquet file.
        """
        start_time = time.time()

        # ensure data directory exists
        os.makedirs("data", exist_ok=True)

        batch_num = 0
        for batch in self.yield_batches(self.tickers, 25):
            print(f"Downloading batch {batch_num}...")

            attempt = 0
            success = False
            while attempt <= retries and not success:
                try:
                    tickers_str = ",".join(batch)
                    df = yf.download(
                        tickers=tickers_str,
                        start=start,
                        end=end,
                        auto_adjust=True,
                        progress=False
                    )["Open"]

                    if df.empty:
                        print(f"⚠️ Batch {batch_num} returned empty, skipping.")
                        break

                    # save each ticker individually
                    for ticker in batch:
                        if ticker in df.columns and not df[ticker].dropna().empty:
                            ticker_df = df[[ticker]].rename(columns={ticker: "Price"})
                            ticker_df.to_parquet(f"data/{ticker}.parquet")
                            print(f"✅ {ticker} parquet created!")

                    success = True  # batch succeeded
                except Exception as e:
                    print(f"⚠️ Issue downloading batch {batch_num}, attempt {attempt+1}: {e}")
                    attempt += 1
                    time.sleep(delay)

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


        total_time = time.time() - start_time
        print(f"Total batching and data loading time: {total_time:0.2f} seconds")

if __name__ == "__main__":
    tickers = get_biggest_losers()
    loader = PriceLoader([tickers])
    loader.save_to_parquet((datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))
