# Todo List

## Cole (Dev)
* Re-run and Save Models on Equity Tickers
* Add Volatility Features (Rolling Window, 3-day)
* WRDS Sentiment Inputs
* Re-train and Save Models
* Subtract SHV
* Save date in readable format and print last date

## Adith (PM)
* Construct Portfolio Allocation & Forward to Jonny

## Jonny (Trader)
* Submit Trading Decisions on CQGOne and Bloomberg

## To-Do: Model Predictions and Backtest Pipeline (Adith)

- [ x ] Build model_loader for each symbol 
- [ x ] Run model and generate prediction CSV on full time series
- [ x ] Load the generated CSV into a `pandas.DataFrame`  
- [ x ] Add two new columns:
  - [ x ] Daily returns for the symbol (from `.parquet`)
  - [ x ] Model signals 
- [ x ] Assign trading signals based on model predictions (`1`, `-1`, `0`)  
- [ x ] Implement a **real backtest**:
  - [ x ] Use signals to compute compound returns (`(1 + ret * signal)`)  
- [ ] Plot total return as a time series graph  