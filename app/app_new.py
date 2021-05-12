from flask import Flask, render_template, request, url_for, flash, redirect, jsonify, session
from werkzeug.exceptions import abort

import json, plotly
import pandas as pd
import datetime as dt

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

    st_data = StockDataAnalysis(ticker, start_date, end_date)
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()

    start_pred = (st_data.end_date + dt.timedelta(days=1))
    end_pred = (st_data.end_date + dt.timedelta(days=7))

    st_model = ModelStockPrice(start_predict=start_pred, end_predict=end_pred)
    st_model.create_train_test_data(st_data, train_size=0.7)
    st_model.fit()
    Y_predict = st_model.predict()
    evaluation_result = st_model.evaluate_model_performance(plot_data=False)


    # Creating the plots for the website
    # 1. create a ploty graph_objs for each plot
        # create a list of lists for x_vals and y_vals...
        # ...if you want multiple data Series to be represented in one plot
    # 2. put all graph_objs into a list (here it's called 'data')
    # 3. create a dictionary for the layout (one layout dict per plot)
    # 4. create a list (here it's called 'figures') which you fill with the 'data' list and the layout each in one dictionary
    g1_trace1 = go.Scatter(
                        x = st_data.data.index,
                        y = st_data.data[ticker],
                        mode = "lines",
                        name = ticker,
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                        text=ticker)

    data1 = [g1_trace1]
    layout1 = dict(title = 'Stock price development (Adj. Close)',
                  xaxis = dict(title= 'time', ticklen= 5, zeroline= False),
                  yaxis = dict(title= 'US$')
                 )

    g2_trace1 = go.Scatter(
                        x = st_data.data_norm.index,
                        y = st_data.data_norm[ticker],
                        mode = "lines",
                        name = ticker,
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                        text=ticker)
    g2_trace2 = go.Scatter(
                        x = st_data.sma.index,
                        y = st_data.sma[ticker],
                        mode = "lines",
                        name = 'SMA',
                        marker = dict(color = 'rgba(0, 75, 175, 1)'),
                        text='daily returns')
    g2_trace3 = go.Scatter(
                        x = st_data.b_upper_band.index,
                        y = st_data.b_upper_band[ticker],
                        mode = "lines",
                        name = 'Upper band',
                        marker = dict(color = 'rgba(202, 123, 87, 0.93)'),
                        text='Upper band')
    g2_trace4 = go.Scatter(
                        x = st_data.b_lower_band.index,
                        y = st_data.b_lower_band[ticker],
                        mode = "lines",
                        name = 'Lower band',
                        marker = dict(color = 'rgba(202, 123, 87, 0.93)'),
                        text='Lower band')

    data2 = [g2_trace1, g2_trace2, g2_trace3, g2_trace4]
    layout2 = dict(title = 'Relative stock price development (Adj. Close)',
                  xaxis = dict(title= 'time', ticklen= 5, zeroline= False)
                 )

    g3_trace1 = go.Scatter(
                        x = st_data.daily_returns.index,
                        y = st_data.daily_returns[ticker],
                        mode = "lines",
                        name = 'Daily returns',
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                        text='Daily returns')

    g3_trace2 = go.Scatter(
                        x = st_data.momentum.index,
                        y = st_data.momentum[ticker],
                        mode = "lines",
                        name = 'Momentum',
                        marker = dict(color = 'rgba(202, 123, 87, 0.93)'),
                        text='Momentum')

    data3 = [g3_trace1, g3_trace2]
    layout3 = dict(title = 'Daily returns & momentum 5 days',
                  xaxis = dict(title= 'time', ticklen= 5, zeroline= False),
                  yaxis = dict(title= '%')
                 )



    figures=[]

    figures.append(dict(data=data1, layout=layout1))
    figures.append(dict(data=data2, layout=layout2))
    figures.append(dict(data=data3, layout=layout3))

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('post.html',
                           ids=ids,
                           figuresJSON=figuresJSON,
                           evaluation_result=evaluation_result)
