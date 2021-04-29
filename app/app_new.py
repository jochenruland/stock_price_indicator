from flask import Flask, render_template, request, url_for, flash, redirect, jsonify, session
from werkzeug.exceptions import abort

import json, plotly
import pandas as pd

from data_wrangling.process_data import StockDataAnalysis
from data_wrangling.train_classifier import ModelStockPrice

# import graph objects as "go"
import plotly.graph_objs as go


app = Flask(__name__)
app.config['SECRET_KEY'] = '928272625242322212'

@app.route('/', methods=('GET', 'POST'))
def index():

    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        if not ticker:
            flash('Ticker symbol is required')
        elif not start_date:
            flash('Start date is required')
        elif not end_date:
            flash('End date is required')
        else:
            session['ticker']=ticker
            session['start_date']=start_date
            session['end_date']=end_date
            return redirect(url_for('post'))

    return render_template('index.html')

@app.route('/post')
def post():
    ticker = session['ticker']
    start_date = session['start_date']
    end_date = session['end_date']

    st_data = StockDataAnalysis([ticker], start_date, end_date, pred_days=7)
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()
    st_data.create_train_test_data(symbol=ticker, train_size=0.8)

    st_model = ModelStockPrice()
    st_model.fit(st_data)
    pred_values = st_model.predict(st_data)

    # Creating the plots for the website
    # 1. create a ploty graph_objs for each plot
        # create a list of lists for x_vals and y_vals...
        # ...if you want multiple data Series to be represented in one plot
    # 2. put all graph_objs into a list (here it's called 'data')
    # 3. create a dictionary for the layout (one layout dict per plot)
    # 4. create a list (here it's called 'figures') which you fill with the 'data' list and the layout each in one dictionary
    trace1 = go.Scatter(
                        x = st_data.data.index,
                        y = st_data.data[ticker],
                        mode = "lines",
                        name = ticker,
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                        text=ticker)

    data = [trace1]
    layout = dict(title = 'Stock price development (Adj. Close)',
                  xaxis= dict(title= 'time', ticklen= 5, zeroline= False)
                 )

    figures=[]

    figures.append(dict(data=data, layout=layout))

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('post.html',
                           ids=ids,
                           figuresJSON=figuresJSON)
