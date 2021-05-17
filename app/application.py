import sys
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from data_wrangling.process_data import StockDataAnalysis
from data_wrangling.train_classifier import ModelStockPrice

def main(symbol='AAPL'):
    print('Enter start date in date format (YYYY-MM-DD):')
    start_date = input()
    sd = dt.datetime.strptime(start_date, '%Y-%m-%d')

    print('Enter end date in date format (YYYY-MM-DD):')
    end_date = input()
    ed = dt.datetime.strptime(end_date, '%Y-%m-%d')

    if start_date >= end_date:
        return print('Start date must be before end date')

    elif (ed - sd).days < 30:
        return print('Timeframe between start date and end date must be minimum 30 days')

    elif ed > dt.datetime.now():
        return print('End date must be equal or before actual date')
    else:
        print('Enter a comma seperated list of ticker symbols (f.ex. AAPL,GOOG,BABA):')
        symbols_str = input()
        symbol_lst = symbols_str.split(",")

    if not symbol_lst:
        return print("No ticker symbol was entered")
    else:
        for symbol in symbol_lst:
            return print(symbol)

    #st_data = StockDataAnalysis(start_date=start_date, end_date=end_date)
    #st_data.setup_features()

    #print(df_indicators.head(50))

    #st_model = ModelStockPrice(start_predict='2021-04-28', end_predict='2021-05-07')
    #st_model.create_train_test_data(st_data, train_size=0.7)
    #st_model.fit()
    #print(st_model.predict())
    #result = st_model.evaluate_model_performance()
    #print(result)

if __name__ == '__main__':
    main()
