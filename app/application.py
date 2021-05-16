import sys
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from data_wrangling.process_data import StockDataAnalysis
from data_wrangling.train_classifier import ModelStockPrice

def main(symbol='AAPL', start_date='2020-11-01', end_date='2021-04-27'):

    st_data = StockDataAnalysis(start_date=start_date, end_date=end_date)
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()
    print(df_indicators.head(50))

    st_model = ModelStockPrice(start_predict='2021-04-28', end_predict='2021-05-07')
    st_model.create_train_test_data(st_data, train_size=0.7)
    st_model.fit()
    print(st_model.predict())
    st_model.evaluate_model_performance()

if __name__ == '__main__':
    main()
