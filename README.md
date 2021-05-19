
# STOCK PRICE Indicator

## Capstone project for Udacity Data Science Nanodegree 2021

This project provides a stock price indicator which downloads historical stock price information from the Yahoo finance platform and performs a technical
analysis calculating a bunch of indicators like the simple moving average, Bollinger bands, momentum, daily returns and cumulative returns. Based on these information the project further provides a machine learning pipeline to predict future stock prices.

Predicting future stock prices is not an easy task. Therefore different machine learning algorithms have been tested first. The application of the different models can be found in the jupyter notebook `Stock_price_indicator.ipynb`. Best results were achieved using Lasso Lars linear regression. Therefore the machine learning pipeline uses this algorithm. But still the quality of the prediction strongly depends on the selected set of historical data. If you train the model with data from a historic time period which includes sudden extreme variations due to the pandamic in March 2020 for example, this will negatively impact the quality of the model. Therefore it is crucial to take the evaluation metrics of the underlying model into account when looking at the predictions. It may be helpful to select a longer or shorter timeframe of historical data if the evaluation of the trained model shows strong deviations and low correlation.   

There are two ways to access the stock price analysis and prediction:
1. Via a flask web app `web_app.py` which launches a web server on `http://127.0.0.1:5000/`
2. Using the scripts `ETL.py` and `ML.py`
    - `ETL.py` provides an interface to enter a time period and a list of stock ticker symbols
      in order to download and analyse the historical data
    - `ML.py` provides an interface to enter a time period and a list of stock ticker symbols
      for which future stock prices shall be predicted. The ticker symbols must be a subset of the
      previously selected stock data

## Installation
Clone this repo to the preferred directory on your computer using `git clone https://github.com/jochenruland/stock_price_indicator`. Then start either the Webapp `/app/web_app.py` or run the two scripts `/app/ETL.py` and `/app/ML.py` .

You must have installed the following libraries to run the code:

`sys`
`pandas`
`numpy`
`matplotlib`
`yfinance`
`datetime`
`sklearn`
`flask`
`werkzeug`
`json`
`plotly`
`sqlite3`

Alternatively you can install `requirements.txt`.

## Program and dataset files:

### Main files
- `app/data_wrangling/process_data.py`: module that includes the class StockDataAnalysis and some helper functions
- `app/data_wrangling/train_classifier.py`: module that includes the class ModelStockPrice
- `app/ETL.py`: ETL pipeline which extracts the data from Yahoo finance, calculates and joins features and safe the result to `indicators.db`
- `app/ML.py`: Machine learning pipeline which loads the stock data from `indicators.db`, instantiates a model object which is then fitted and used for prediction
- `app/web_app.py`: Starts the Python server for the Webapp.

@WORK---------------------------
### File structure
```
app
|- web_app.py # Flask file that runs app
|- ETL.py
|- ML.py
|- indicators.db # Database saving results from ETL.py
|- templates
| |- index.html # main page of web app
| |- base.html # formatting page of web app
| |- post.html # result page of web app
|- data_wrangling
| |- process_data.py
| |- train_classifier.py
README.md
LICENSE.txt
requirements.txt
Stock_price_indicator.ipynb
```
