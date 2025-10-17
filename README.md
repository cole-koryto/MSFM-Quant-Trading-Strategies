# MSFM-Quant-Trading-Strategies

> Collection of quantitative trading strategies, forecasting models, and a lightweight backtesting / experiment harness.

This repository contains Python code for strategy implementation, prediction/forecasting, a small backtesting engine, and orchestration scripts for experiments. It was developed as part of the MSFM (Master of Science in Financial Mathematics) coursework/research.


## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/cole-koryto/MSFM-Quant-Trading-Strategies.git
cd MSFM-Quant-Trading-Strategies
````

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
# or on Mac
pip install -r requirements-mac.txt
```

---

## File Descriptions

* **`backtester.py`** – Step through time, generate orders, track cash/positions, compute P&L and performance metrics (returns, drawdown, Sharpe).
* **`strategy.py`** – Base strategy interface, helper functions, and indicator calculations.
* **`dollar_neutral_strategy.py`** – Example dollar-neutral strategy (long/short net neutral exposure).
* **`forecaster.py`** – File that contains all of the code to build, test, and run the LSTM and XGBoost models.
* **`predictor.py`** – Convert model predictions into actionable signals.
* **`model_loader.py`** – Load serialized or saved models (TensorFlow / XGBoost / pickle).
* **`price_loader.py`** – Gathers data from Yahoo finance and saves to parquets.
* **`main.py`** – Script to run some model on the entire dataset available and save best model
* * **`main_testing.py`** – Scripts for running experiments and test runs. Also the finds best hyperparameters for XGBoost.
* **`requirement.txt` / `requirements-mac.txt`** – Python dependency specifications.
* **`todo.md`** – Project TODO list and near-term improvements.

---

## Requirements
* Windows requirements in requirement.txt
* For mac compatibility, MUST use Python 3.11-
* Tensorflow 2.15.0 works for mac
