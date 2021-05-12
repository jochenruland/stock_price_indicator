import sys
from process_data import StockDataAnalysis
import datetime as dt
import numpy as np
import matplotlib as plt

from sklearn import preprocessing
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

class ModelStockPrice():
    def __init__(self, start_predict=None, end_predict=None):
        '''Create an instance of the model to predict stockprice'''

        self.start_predict = start_predict
        self.end_predict = end_predict

        self.model = linear_model.LassoLars(alpha = 0.1)


    def create_train_test_data(self, stockdata, train_size=0.8):
        ''' Splits the indicator dataframe into a train and test dataset and standardizes the data of the indipendent variable
            INPUT:
            indicator_df - dataframe object - dataframe which contains the Adj Close and different indicators for each symbol
            symbol - str - symbol of the listed company for which you want to predict stock price
            train_size - float - size of train dataset
            start_pred - str - start date of prediction
            end_pred - str - end date of prediction
            OUTPUT:
            pred_days - int - days to be predicted
            Y_train - 1d array - contains the training dataset of the dependent variable (stock price)
            Y_test - 1d array - contains the test dataset of the dependent variable (stock price)
            X_train - nd array - contains the training dataset of the independent variables
            X_test - nd array - contains the test dataset of the independent variables
            X_predict = nd array - contains the data of the independent variable for the prediction period
            time_series_train - 1d array - selected time period of training data
            time_series_test - 1d array - selected time period of test data
            time_series_test - 1d array - predicted time period

        '''
        sd = dt.datetime.strptime(self.start_predict, '%Y-%m-%d')
        ed = dt.datetime.strptime(self.end_predict, '%Y-%m-%d')

        try:
            if sd >= ed:
                raise ValueError('Start date beyound end date')
            else:
                self.pred_days = (ed-sd).days

                indicator_df = stockdata.indicator_df[stockdata.indicator_df['Date'] <= self.start_predict]

                df = stockdata.indicator_df.copy().drop(['Symbol','Date'], axis=1)

                for i in range(1, self.pred_days):
                    indicator_df=indicator_df.join(df.shift(i), rsuffix="[{} day before]".format(i))


                train_df = indicator_df.copy().iloc[self.pred_days:] # Training data starts from the date where data for all indicators is available

                if self.pred_days > 0:
                    X = train_df.iloc[:-self.pred_days,3:] # Reduces the X Date by the number of pred_days at the end of the dataframe
                    self.X_predict = preprocessing.scale(train_df.iloc[-self.pred_days:, 3:])
                    Y = train_df.drop('Symbol', axis=1).iloc[self.pred_days:,:2] # Starts at pred_days and takes all data until the end of the dataframe

                    X.fillna(method='ffill', inplace=True)
                    X.fillna(method='bfill', inplace=True)

                    Y.fillna(method='ffill', inplace=True)
                    Y.fillna(method='bfill', inplace=True)


                train_ct = int(X.shape[0] * train_size)
                test_ct = X.shape[0] - train_size

                self.X_train, self.X_test = preprocessing.scale(X.iloc[:train_ct]), preprocessing.scale(X.iloc[train_ct:])
                self.Y_train, self.Y_test = Y.iloc[:train_ct]['Adj Close'].copy().tolist(), Y.iloc[train_ct:]['Adj Close'].copy().tolist()

                self.time_series_train = Y.iloc[:train_ct].Date
                self.time_series_test = Y.iloc[train_ct:].Date


                return self.pred_days, self.X_train, self.Y_train, self.X_test, self.Y_test, self.time_series_train, self.time_series_test, self.X_predict


        except ValueError:
            raise


    def fit(self):
        '''Fit the model with training data '''
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        '''Predict stockprice '''
        self.Y_predict = self.model.predict(self.X_test)
        self.Y_future = self.model.predict(self.X_predict)
        return self.Y_predict, self.Y_future

    def evaluate_model_performance(self):
        '''Evaluate prediction'''

        rmse = np.sqrt(np.sum((self.Y_test - self.Y_predict) **2)/len(self.Y_predict)) #(root mean squared error)
        mse = mean_squared_error(self.Y_test, self.Y_predict)
        corr = np.corrcoef(self.Y_test, self.Y_predict)
        corrcoef = corr[0,1]
        mae = mean_absolute_error(self.Y_test, self.Y_predict)
        mape = mean_absolute_percentage_error(self.Y_test, self.Y_predict)
        r2 = r2_score(self.Y_test, self.Y_predict)

        fig = plt.figure(figsize=(12,8))

        value_days = len(self.Y_future)
        end_date = (self.time_series_test.iloc[-1] + dt.timedelta(days=value_days+1))

        time_series_future = pd.date_range(self.time_series_test.iloc[-1]+ dt.timedelta(days=2) , end_date).tolist()

        plt.plot(self.time_series_test, self.Y_test, color='lightblue', linewidth=2, label='test data')
        plt.plot(self.time_series_test, self.Y_predict.reshape(-1,1), color='red',  linewidth=2, label='predicted data')

        plt.plot(time_series_future , self.Y_future.reshape(-1,1), color='green',  linewidth=2, label='future predicted data')

        plt.legend()
        plt.show()

        return print(('RMSE {} \n MSE {} \n MAE {} \n MAPE {} \n r2 {} \n CORRCOEF {} \n').format(rmse, mse, mae, mape, r2 , corrcoef))



def main(symbol='AAPL', start_date='2020-01-01', end_date='2021-04-27'):

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
