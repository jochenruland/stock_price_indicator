import sys
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

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
        symbol_lst = symbols_str.strip(' ').split(",")

    if not symbol_lst:
        return print("No ticker symbol was entered")
    else:
        data_lst=[]
        for s in symbol_lst:
            s=s.upper()
            st_data = StockDataAnalysis(symbol=s,start_date=sd, end_date=ed)
            st_data.setup_features()
            df_indicators = st_data.create_indicator_dataframe()
            data_lst.append(dict(symbol = s, data = df_indicators ))
            conn = sqlite3.connect('indicators.db')
            df_indicators.to_sql('s', con = conn, if_exists='replace', index=False)

        print(data_lst)


    #print('Enter a comma seperated list of ticker symbols, \n you want to make predictions for (all or subset of previous list):')
    #pred_str = input()
    #pred_lst = pred_str.strip(' ').split(",")

    #print('Enter a start date for predictions:')


    #st_model = ModelStockPrice(start_predict='2021-04-28', end_predict='2021-05-07')
    #st_model.create_train_test_data(st_data, train_size=0.7)
    #st_model.fit()
    #print(st_model.predict())
    #result = st_model.evaluate_model_performance()
    #print(result)

if __name__ == '__main__':
    main()
