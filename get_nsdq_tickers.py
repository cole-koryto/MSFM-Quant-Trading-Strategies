import pandas as pd
import yfinance as yf
import time
import requests
import io
import os

@staticmethod
def get_nasdaq_100():
    # get all S&P tickers from today, use headers to avoid 403 Forbidden Errors
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    # Use Wikipedia's S&P 500 list to get all relevant S&P 500 companies
    response = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers)

    # ensure response was successful
    if response.status_code == 200:
        # wrap response's text in io (deprecation issue), then get nested pandas df (index 0), ticker column only
        tables = pd.read_html(io.StringIO(response.text))
    else:
        print(f"Request failed with status code: {response.status_code}")[4]["Ticker"]
    # correct table
    tickers = list(tables[4]["Ticker"].values)
    return tickers

print(get_nasdaq_100())