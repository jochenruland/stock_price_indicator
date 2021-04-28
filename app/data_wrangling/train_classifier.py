import sys
from process_data import StockDataAnalysis

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class ModelStockPrice():
    def __init__(self):
        '''Create an instance of the model to predict stockprice'''
        self.knn = KNeighborsRegressor(weights='distance')

        # self.model = AdaBoostRegressor(base_estimator=self.knn, random_state=42)
        self.model = RandomForestRegressor(random_state=42, criterion='mae',
                                             n_estimators=30, min_samples_split=5)

    def fit(self, stockdata):
        '''Fit the model with training data '''
        self.model.fit(stockdata.X_train, stockdata.Y_train)

    def predict(self, stockdata):
        '''Predict stockprice '''
        return self.model.predict(stockdata.X_pred)

    def evaluate_model_performance(self, stockdata, Y_pred):
        '''Evaluate prediction'''
        rmse = np.sqrt(np.sum((stockdata.Y_test - Y_predict) **2)/len(Y_predict)) #(root mean squared error)
        corr = np.corrcoef(x=stockdata.Y_test, y=Y_predict)

        fig = plt.figure(figsize=(12,8))

        plt.plot(stockdata.time_series_test, stockdata.Y_test, color='lightblue', linewidth=2, label='test data')
        plt.plot(stockdata.time_series_test, Y_pred, color='red',  linewidth=2, label='predicted data')
        plt.legend()

        return rmse, corr


def main(symbols=['AAPL','GOLD','FB'], start_date='2006-01-01', end_date='2021-04-27'):

    st_data = StockDataAnalysis(symbols, start_date, end_date, pred_days=7)
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()
    st_data.create_train_test_data(symbol='GOLD', train_size=0.8)

    st_model = ModelStockPrice()
    st_model.fit(st_data)
    print(st_model.predict(st_data))


'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
'''


if __name__ == '__main__':
    main()
