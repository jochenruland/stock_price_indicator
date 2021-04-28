# import libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn import preprocessing

# Import data from Yahoo finance

def get_data(symbols=['SPY'], start_date='2020-01-01', end_date='2020-12-31'):
    '''
    Setup of an empty dataframe with the given timeperiod as index to be used as instance for further gathered data.
    Then downloads the data from Yahoo Finance for the selected symbol(s) and time period and selects the Adj Close column
    INPUT:
    symbols - list - symbols of listed stocks
    start_date - datetime - Beginning of the period to analyze
    end_date - datetime - End of the period to analyze

    OUTPUT
    df - dataframe - Dataframe containing the Adj Close for each symbol with the time period as index (ordered ascending)
    '''
    dates= pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_tmp = yf.download(symbol, start_date, end_date)
        df_tmp = df_tmp[['Adj Close']]
        df_tmp = df_tmp.rename(columns={"Adj Close": symbol})

        df = df.join(df_tmp)
        df = df.dropna()

    return df


# Noramlize the stock price data
def normalize_stock_data(df):
    df = df/df.iloc[0,:]
    return df


class StockDataAnalysis():
    ''' Creates a StockDataAnalysis object which is able to take one or mutiple stock symbols and a timeframe and then computes
        a range of indicators on the stock data and plots the results'''

    def __init__(self, symbols=['SPY'], start_date='2020-01-01', end_date='2020-12-31', pred_days=7):
        ''' Create an instance of StockDataAnalysis'''
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.pred_days = pred_days

        self.data = get_data(self.symbols, self.start_date, self.end_date)
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
            ax2.set_ylabel('price')
            ax2.set_title(title_str)
            for col in df.columns:
                df[col].plot()

            plt.legend(loc='upper right')
            plt.show()

    # Calcuate different features
    def calculate_rolling_stats(self, win=20):
        rm = self.data_norm.rolling(window=win).mean()
        rstd = self.data_norm.rolling(window=win).std()
        self.sma = rm.dropna()
        self.rstd = rstd.dropna()

    def calculate_bollinger_bands(self):
        self.upper_band = self.sma + self.rstd*2
        self.lower_band = self.sma - self.rstd*2

    def calculate_daily_returns(self):
        daily_returns = self.data.copy()
        daily_returns[1:] = (self.data[1:]/self.data[:-1].values) - 1
        daily_returns.iloc[0,:] = 0
        self.daily_returns = daily_returns

    def calculate_momentum(self, win=5):
        self.momentum = self.data.copy()
        self.momentum[win:] = (self.data[win:]/self.data[:-(win)].values) - 1
        self.momentum.iloc[0:(win),:] = 0

    def setup_features(self):
        self.calculate_rolling_stats()
        self.calculate_bollinger_bands()
        self.calculate_daily_returns()
        self.calculate_momentum()

    # Setup a joint dataframe including all calculated features
    def create_indicator_dataframe(self):
        ''' Function which which takes the Adj Close and corresponding dates per symbol, adds a new column containing the symbol
            and joins all indicators to one dataframe
            INPUT:
            df - dataframe - contains the orginal data to analyse
            OUTPUT:
            indicator_df - dataframe - contains the Adj Close and all indicators as features tagged by the symbol '''

        self.indicator_df = pd.DataFrame(columns=['Date','Symbol', 'Adj Close','Daily Returns','SMA','Momentum','Upper Band','Lower Band'])

        for symbol in self.data.columns:
            df_temp = self.data[symbol].reset_index().rename(columns={'index':'Date', symbol:'Adj Close'})
            df_temp['Symbol'] = symbol
            df_temp = df_temp.join(self.daily_returns[symbol], on='Date').rename(columns={symbol:'Daily Returns'})
            df_temp = df_temp.join(self.sma[symbol], on='Date').rename(columns={symbol:'SMA'})
            df_temp = df_temp.join(self.upper_band[symbol], on='Date').rename(columns={symbol:'Upper Band'})
            df_temp = df_temp.join(self.lower_band[symbol], on='Date').rename(columns={symbol:'Lower Band'})
            df_temp = df_temp.join(self.momentum[symbol], on='Date').rename(columns={symbol:'Momentum'})

            self.indicator_df = pd.concat([self.indicator_df, df_temp])

            self.indicator_df.fillna(method='ffill', inplace=True)
            self.indicator_df.fillna(method='bfill', inplace=True)

        return self.indicator_df

    def create_train_test_data(self, symbol='SPY', train_size=0.8):
        ''' Splits the indicator dataframe into a train and test dataset and standardizes the data of the indipendent variable
            INPUT:
            indicator_df - dataframe object - dataframe which contains the Adj Close and different indicators for each symbol
            symbol - str - symbol of the listed company for which you want to predict stock price
            train_size - float - size of train dataset
            OUTPUT:
            Y_train - 1d array - contains the training dataset of the dependent variable (stock price)
            Y_test - 1d array - contains the test dataset of the dependent variable (stock price)
            X_train - nd array - contains the training dataset of the independent variables
            X_test - nd array - contains the test dataset of the independent variables
            time_series_train - 1d array - selected time period of training data
            time_series_test - 1d array - selected time period of test data
        '''
        train_data = int(self.indicator_df[self.indicator_df['Symbol']==symbol].shape[0] * train_size)
        test_size = self.indicator_df[self.indicator_df['Symbol']==symbol].shape[0] - train_data

        self.X_train = preprocessing.scale(self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[20:train_data,3:9])
        self.X_test = preprocessing.scale(self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[train_data:,3:9])
        self.X_pred = preprocessing.scale(self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[-(self.pred_days):,3:9])
        self.Y_train = self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[20:train_data,2].values
        self.Y_test = self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[train_data:,2].values
        self.time_series_train = self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[20:train_data,0].values
        self.time_series_test = self.indicator_df[self.indicator_df['Symbol']==symbol].iloc[train_data:,0].values



def main(symbols=['AAPL'], start_date='2020-01-01', end_date='2020-12-31'):
    ''' This Function checks the class StockDataAnalysis '''

    st_data = StockDataAnalysis(symbols, start_date, end_date)
    st_data.plot_stock_data()
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()
    print(df_indicators.head(50))

if __name__ == '__main__':
    main()
