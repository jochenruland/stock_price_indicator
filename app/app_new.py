from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from werkzeug.exceptions import abort

import json
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
            return redirect(url_for('post'))

    return render_template('index.html')

@app.route('/post')
def post():
    return render_template('post.html')


'''
    post = get_post(post_id)

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    st_data = StockDataAnalysis([ticker], start_date, end_date, pred_days=7)
    st_data.setup_features()
    df_indicators = st_data.create_indicator_dataframe()
    st_data.create_train_test_data(symbol=ticker, train_size=0.8)

    st_model = ModelStockPrice()
    st_model.fit(st_data)
    print(st_model.predict(st_data))


#-----------------------------------
    # Creating trace1
    trace1 = go.Scatter(
                        x = df.world_rank,
                        y = df.citations,
                        mode = "lines",
                        name = "citations",
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                        text= df.university_name)
    # Creating trace2
    trace2 = go.Scatter(
                        x = df.world_rank,
                        y = df.teaching,
                        mode = "lines+markers",
                        name = "teaching",
                        marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                        text= df.university_name)
    data = [trace1, trace2]
    layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
                  xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)

#------------------------------------------------

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



    return render_template('post.html', post=post)



#------------------------------------------------------------------

@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO posts (title, content) VALUES (?, ?)',
                         (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('create.html')

@app.route('/<int:id>/edit', methods=('GET', 'POST'))
def edit(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE posts SET title = ?, content = ?'
                         ' WHERE id = ?',
                         (title, content, id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('edit.html', post=post)

@app.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post['title']))
    return redirect(url_for('index'))
'''
