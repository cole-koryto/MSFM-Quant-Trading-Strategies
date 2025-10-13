from forecaster import Forecaster
import pandas as pd
import os
import datetime


def main():
    # Makes a prediction of next day of each symbol
    symbols = ["ES=F", "^SPX"]
    results = []
    for symbol in symbols:
        forecaster = Forecaster(symbol=symbol)
        df_pred = forecaster.run_LSTM(lags=30)
        results.append(df_pred)

    # Combine all predictions into one table
    final_df = pd.concat(results, ignore_index=True)

    # Create folder if it doesn't exist
    os.makedirs("predictions", exist_ok=True)

    # Create filename with symbol + date
    today = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"predictions/asset-predictions{today}.csv"

    # Save predictions to CSV
    final_df.to_csv(filename, index=False)
    print(f"✅ Saved predictions to {filename}")


if __name__ == "__main__":
    main()