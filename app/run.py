import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # visual1 for category counts
    category_name = df.iloc[:,4:].sum()
    category_name = category_name.sort_values(ascending=False)
    category = list(category_name.index)
    
    # visual2 for distribution of message categories
    category_cols = df.iloc[:,4:].columns
    category_vals = (df.iloc[:,4:] !=0).sum().values
    
    graphs = [
         # visual1 for distribution of message categories
        {
            'data': [
                Bar(
                    x=category_cols,
                    y=category_vals
                )
            ],

            'layout': {
                'title': 'Message Category Distribution',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 30
                }
            }
        },
        # visual2 for category counts
        {
            'data': [
                Bar(
                    x=category,
                    y=category_name
                )
            ],

            'layout': {
                'title': 'Message Category Counts',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 25
                }
            }
        },
       # example visual for the distribution of message genre
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
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()