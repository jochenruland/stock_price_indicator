
# STOCK PRICE Indicator

## Capstone project for Udacity Data Science Nanodegree 2021

This project provides a stock price indicator which downloads historical stock price information from the Alpha-Vantage platform. In order to download data from Alpha-Vantage please refer to https://www.alphavantage.co/support/#api-key. The API key has to be set in `process_data.py` for the parameter `A_key`. This version uses my personal API key which will be deactivated after review of the project.

Once the data has been downloaded the application performs a technical analysis calculating a bunch of indicators like the simple moving average, Bollinger bands, momentum, daily returns and cumulative returns. Based on these information the project further provides a machine learning pipeline to predict future stock prices.

Different machine learning algorithms have been tested first. The application of the different models can be found in the Jupyter notebook `Stock_price_indicator.ipynb`. Once everything has been analyzed and tested in the Jupyter Notebook the final application has been implemented in python scripts and a Webapp. For a detailed description of the project please refer to `Project_documentation_20210531.pdf`    

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
`alpha_vantage.timeseries`
`datetime`
`sklearn`
`flask`
`werkzeug`
`json`
`plotly`
`sqlite3`

## Program and dataset files:

### Main files
- `app/data_wrangling/process_data.py`: module that includes the class StockDataAnalysis and some helper functions
- `app/data_wrangling/train_classifier.py`: module that includes the class ModelStockPrice
- `app/ETL.py`: ETL pipeline which extracts the data from Yahoo finance, calculates and joins features and safes the result to `indicators.db`
- `app/ML.py`: Machine learning pipeline which loads the stock data from `indicators.db`, instantiates a model object which is then fitted and used for prediction
- `app/web_app.py`: Starts the Python server for the Webapp.

### File structure
```
app
|- web_app.py # Flask file that runs app
|- ETL.py
|- ML.py
|- indicators.db # Database saving results from ETL.py
|- AAPL.csv # Sample data download for Apple
|- BABA.csv # Sample data download for Alibaba
|- GOLD.csv # Sample data download for Barrick Gold
|- GOOG.csv # Sample data download for Alphabet
|- SPY.csv  # Sample data download for S&P 500 Market Index
|- templates
| |- index.html # main page of web app
| |- base.html # formatting page of web app
| |- post.html # result page of web app
|- data_wrangling
| |- process_data.py
| |- train_classifier.py
README.md
Project_documentation_20210531
LICENSE.txt
Stock_price_indicator.ipynb
```
