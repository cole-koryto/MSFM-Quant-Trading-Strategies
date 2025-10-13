from forecaster import Forecaster


def main():
    forecaster = Forecaster()
    forecaster.test_XGBoost(30, 180)


if __name__ == "__main__":
    main()