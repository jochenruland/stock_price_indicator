import sys
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

from data_wrangling.process_data import StockDataAnalysis
from data_wrangling.train_classifier import ModelStockPrice

def main(symbol='AAPL'):
    print('Enter start date for prediction (YYYY-MM-DD):')
    start_date = input()
    sd = dt.datetime.strptime(start_date, '%Y-%m-%d')

    print('Enter end date for prediction (YYYY-MM-DD):')
    end_date = input()
    ed = dt.datetime.strptime(end_date, '%Y-%m-%d')

    if sd >= ed:
        return print('Start date must be before end date')

    elif (ed - sd).days > 7:
        return print('No predictions for more than 7 days will be made')

    #elif sd > dt.datetime.now():
        #return print('End date must be equal or before actual date')
    else:
        print('Enter a comma seperated list of symbols to make predticions for. \
            \n Data must have been selected via ETY.py previously:')
        symbols_str = input()
        symbol_lst = symbols_str.replace(" ", "").split(",")

    if not symbol_lst:
        return print("No ticker symbol was entered")
    else:
        conn = sqlite3.connect('indicators.db')
        for s in symbol_lst:
            s=s.upper()
            indicators_df = pd.read_sql('SELECT * FROM {}'.format(s), conn)
            print(indicators_df.head())

    st_model = ModelStockPrice(start_predict=sd, end_predict=ed)
    #st_model.create_train_test_data(st_data, train_size=0.7)
    #st_model.fit()
    #print(st_model.predict())
    #result = st_model.evaluate_model_performance()
    #print(result)

if __name__ == '__main__':
    main()
