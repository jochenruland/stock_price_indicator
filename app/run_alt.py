import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from ETL.get_data import #Function to get the data


app = Flask(__name__)

# load data
### engine = create_engine('sqlite:///../data/YourDatabaseName.db')
### df = pd.read_sql_table('YourTableName', engine)

# load model
### model = joblib.load("../models/your_model_name.pkl")


# index webpage receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
'''
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('index.html', ids=ids, graphJSON=graphJSON)
'''
    query = request.args.get('query', '')
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    #query = request.args.get('query', '')

    # use model to predict stockdata for query

    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        regression_result=regression_results
        #classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
