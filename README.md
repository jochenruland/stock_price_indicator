
# STOCK PRICE Indicator
Capstone project for Udacity Data Science Nanodegree 2021

This Project provides a stock price indicator which downloads historical stock price information from the Yahoo finance platform and performs a technical
analysis calculating a bunch of indicators like the simple moving average, Bollinger bands, momentum, daily returns and cumulative returns.Based on these information the project further provides a machine learning pipeline to predict future stock prices.

There are two ways to access the stock price analysis and prediction:
1. Via a flask web app `web_app.py` which launches a web server on `http://127.0.0.1:5000/`
2. Using the scripts `ETL.py` and `ML.py`
    - `ETL.py` provides an interface to enter a time period and a list of stock ticker symbols
      in order to download and analyse the historical data
    - `ML.py` provides an interface to enter a time period and a list of stock ticker symbols
      for which future stock prices shall be predicted. The ticker symbols must be a subset of the
      previously selected stock data

@WORK
Usually thousands of messages from different sources like Email, SMS, Twitter Feeds, etc. will reach people or organizations trying to help in case of a disaster while time is the most crucial resource. In this case this application intends to support these people and organizations by analyzing and automatically classifying the incoming messages and thus being able in a better way to allocate scarce resources.  

The project consists of 3 main pillars:
1. The file `process_data.py` contains an ETL Pipeline to extract the data from 2 csv files, transform the data to one dataframe and load the result into a SQLite database file.

2. The file `train_classifier.py` contains a ML Pipeline which loads the data from the SQLite database, splits up the category column and transforms the categories into binary values. The file further contains a custom tokenizer to normalize, tokanize, lemmantize and stem the messages text in preparation for use in the machine learning model. The ML model uses `CountVectorizer(), TfidfTransformer()` and `MultiOutputClassifier(RandomForestClassifier()` as well as `GridSearchCV` to process the text messages and classify them into 36 categories. The trained model is finally saved in a pickle file named `classifier.pkl`.

3. The file `run.py` starts the Flask Webapp and prepares 3 visualizations which are then displayed on the frontend.

## Installation
Clone this repo to the preferred directory on your computer using `git clone https://github.com/jochenruland/disaster_response_pipeline_project`. The file `/app/run.py` starts the Webapp.

### Libraries
You must have installed the following libraries to run the code:
`pandas`
`numpy`
`re`
`chardet`
`sqlalchemy`
`nltk`
`sklearn`
`pickle`
`json`
`plotly`
`flask`
`joblib`

### Program and dataset files:

### MAIN FILES
- `data/process_data.py`: The ETL pipeline used to extract, load and transform the data needed for model building.
- `data/DisasterResponse.db`: SQLite database file where the result from the ETL pipeline is saved.
- `models/train_classifier.py`: The Machine Learning pipeline used to train and test the model, and evaluate its results. The model is saved as `classifier.pkl`.
- `app/run.py`: Starts the Python server for the Webapp.

### FILE STRUCTURE
```
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md
```

### Instructions to run the application:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/


## License
The MIT License (MIT)

Copyright (c) 2021 Jochen Ruland

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
