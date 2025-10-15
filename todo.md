# Todo List

## Cole (Dev)
* Re-run and Save Models on Equity Tickers
* Add Volatility Features (Rolling Window, 3-day)
* WRDS Sentiment Inputs
* Re-train and Save Models
* Subtract SHV
* Save date in readable format and print last date

## Adith (PM)
* Trading Strat Backtest Code (Performance Metrics and Risk Metrics)
* Construct Portfolio Allocation & Forward to Jonny

## Jonny (Trader)
* Help Adith with Backtest Code
* Submit Trading Decisions on CQGOne and Bloomberg

## To-Do: Model Predictions and Backtest Pipeline

- [ ] Run model on full time series and generate prediction CSV  
- [ ] Load the generated CSV into a `pandas.DataFrame`  
- [ ] Add two new columns:
  - [ ] Daily returns for the symbol (from `.parquet`)
  - [ ] Model signals  
- [ ] Assign trading signals based on model predictions (`1`, `-1`, `0`)  
- [ ] Create a verbose signal column with labels: `"BUY"`, `"SELL"`, `"HOLD"`  
- [ ] Add a new column for **total return**  
- [ ] Implement a **real backtest**:
  - [ ] Use signals to compute compound returns (`(1 + ret * signal)`)  
- [ ] Plot total return as a time series graph  