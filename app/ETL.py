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
        for s in symbol_lst:
            s=s.upper()
            st_data = StockDataAnalysis(symbol=s,start_date=sd, end_date=ed)
            if st_data.data.shape[0] == 0:
                print('No data could be found for ticker symbol {}'.format(s))
            else
                st_data.setup_features()
                df_indicators = st_data.create_indicator_dataframe()
                conn = sqlite3.connect('indicators.db')
                df_indicators.to_sql('s', con = conn, if_exists='replace', index=False)
                print('Stock data for {} has been saved to indicators.db'.format(s))

if __name__ == '__main__':
    main()
