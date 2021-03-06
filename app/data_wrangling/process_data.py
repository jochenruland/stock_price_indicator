# import libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from alpha_vantage.timeseries import TimeSeries
import os.path

# Download data from Alpha-Vantage
A_key = 'AG0F6BKQTUPYSY99'


def download_stock_data(symbol, API_key= A_key):
    try:
        ts = TimeSeries(key = API_key, output_format = 'pandas')
        data = ts.get_daily_adjusted(symbol, outputsize='full')
        symbol = symbol.upper()
        df = data[0]['5. adjusted close'].reset_index().rename(columns={"5. adjusted close": symbol}).sort_values('date', ascending =True)
        df.to_csv(symbol+'.csv', index = False)
        return df
    except Exception as e:
                print('Execption occurred: {}'.format(e))

# Import data from csv file

def get_data(symbol='AAPL', start_date='2020-01-01', end_date='2020-12-31'):
    '''
    Setup an empty dataframe with the given timeperiod as index to be used as instance for further gathered data.
    Then loads data from a .csv file for the selected symbol and selects the Adj Close column
    INPUT:
    symbols - list - symbols of listed stocks
    start_date - datetime - Beginning of the period to analyze
    end_date - datetime - End of the period to analyze

    OUTPUT
    df - dataframe - Dataframe containing the Adj Close for the symbol with the time period as index (ordered ascending)
    '''
    try:
        dates= pd.date_range(start_date, end_date)
        df = pd.DataFrame(index=dates)

        if os.path.isfile(symbol+'.csv') == False:
            print("No such file exists; will be downloaded")
            download_stock_data(symbol)

        df_tmp = pd.read_csv(symbol+'.csv', index_col = 'date')

        df = df.join(df_tmp)
        df = df.dropna()

        return df
    except Exception as e:
                print('Execption occurred: {}'.format(e))

# Noramlize the stock price data
def normalize_stock_data(df):
    df = df/df.iloc[0,:]
    return df


class StockDataAnalysis():
    ''' Creates a StockDataAnalysis object which is able to take one or mutiple stock symbols and a timeframe and then computes
        a range of indicators on the stock data and plots the results'''

    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date='2021-04-16'):
        ''' Create an instance of StockDataAnalysis'''
        self.symbol = symbol

        if isinstance(start_date, str):
            self.start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.start_date = start_date
            self.end_date = end_date

        self.data = get_data(self.symbol, self.start_date, self.end_date)
        if self.data.shape[0] != 0:
            self.data_norm = normalize_stock_data(self.data)

    # Plot stock price data and check for anomalies

    def plot_stock_data(self, normalized=True):
        if normalized:
            df = self.data_norm
            title_str = 'Relative stock price development'
        else:
            df = self.data
            title_str = 'Absolute stock price development'
        if isinstance(df, pd.Series):
            plt.figure(figsize=(12,8))
            ax1 = df.plot()
            ax1.set_xlabel('time')
            ax1.set_ylabel('price')
            ax1.set_title(title_str)
            plt.legend(loc='upper right')
            plt.show()
        else:
            plt.figure(figsize=(12,18))
            ax2 = plt.subplot(2,1,1)
            ax2.set_xlabel('time')
            ax2.set_ylabel('price development')
            ax2.set_title(title_str)
            for col in df.columns:
                df[col].plot()
            plt.legend(loc='upper right')
            plt.show()

    def calculate_rolling_stats(self, win=10):
        rm = self.data_norm.rolling(window=win).mean()
        rstd = self.data_norm.rolling(window=win).std()
        self.sma = rm.dropna()
        self.rstd = rstd.dropna()

    def calculate_bollinger_bands(self):
        self.b_upper_band = self.sma + self.rstd*2
        self.b_lower_band = self.sma - self.rstd*2

    def calculate_daily_returns(self):
        daily_returns = self.data.copy()
        daily_returns[1:] = (self.data[1:]/self.data[:-1].values) - 1
        daily_returns.iloc[0,:] = 0
        self.daily_returns = daily_returns


    def calculate_cumulative_returns(self):
        cumulative_returns = self.data.copy
        cumulative_returns= (self.data/self.data.iloc[0]) - 1
        self.cumulative_returns = cumulative_returns


    def calculate_momentum(self, win=5):
        self.momentum = self.data.copy()
        self.momentum[win:] = (self.data[win:]/self.data[:-(win)].values) - 1
        self.momentum.iloc[0:(win),:] = 0


    def get_market_index(self, market_ix='SPY'):
        self.market_ix = market_ix
        self.market_index = get_data(symbol=market_ix, start_date=self.start_date, end_date=self.end_date)

    def setup_features(self, market_ix='SPY'):
        self.calculate_rolling_stats()
        self.calculate_bollinger_bands()
        self.calculate_daily_returns()
        self.calculate_cumulative_returns()
        self.calculate_momentum()
        self.get_market_index(market_ix=market_ix)


    def create_indicator_dataframe(self):
        ''' Function which which takes the Adj Close and corresponding dates per symbol, adds a new column containing the symbol
            and joins all indicators to one dataframe
            INPUT:
            object
            OUTPUT:
            indicator_df - dataframe - contains the Adj Close and all indicators as features tagged by the symbol '''

        self.indicator_df = pd.DataFrame(columns=['Date','Symbol', 'Adj Close','Daily Returns','Cumulative Returns','SMA', 'Momentum', 'Upper Band','Lower Band','Market Index'])

        for symbol in self.data.columns:
            df_temp = self.data[symbol].reset_index().rename(columns={'index':'Date', symbol:'Adj Close'})
            df_temp['Symbol'] = symbol

            df_temp = df_temp.join(self.daily_returns[symbol], on='Date').rename(columns={symbol:'Daily Returns'})
            df_temp = df_temp.join(self.cumulative_returns[symbol], on='Date').rename(columns={symbol:'Cumulative Returns'})
            df_temp = df_temp.join(self.sma[symbol], on='Date').rename(columns={symbol:'SMA'})
            df_temp = df_temp.join(self.momentum[symbol], on='Date').rename(columns={symbol:'Momentum'})
            df_temp = df_temp.join(self.b_upper_band[symbol], on='Date').rename(columns={symbol:'Upper Band'})
            df_temp = df_temp.join(self.b_lower_band[symbol], on='Date').rename(columns={symbol:'Lower Band'})
            df_temp = df_temp.join(self.market_index[self.market_ix], on='Date').rename(columns={self.market_ix:'Market Index'})

            self.indicator_df = pd.concat([self.indicator_df, df_temp])

            self.indicator_df.fillna(method='ffill', inplace=True)
            self.indicator_df.fillna(method='bfill', inplace=True)
            self.indicator_df.dropna()

        return self.indicator_df

def main(symbol='APPL', start_date='2020-01-01', end_date='2020-12-31'):
    ''' This Function creates an instance of the StockDataAnalysis class and plots the result '''
    try:
        st_data = StockDataAnalysis(symbol=symbol, start_date=start_date, end_date=end_date)
        st_data.setup_features()
        df_indicators = st_data.create_indicator_dataframe()
        print(df_indicators.head(50))
        st_data.plot_stock_data(normalized=False)
    except Exception as e:
        print('Execption occurred: {}'.format(e))

if __name__ == '__main__':
    main()
