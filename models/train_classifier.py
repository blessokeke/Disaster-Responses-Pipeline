import sys
import os
import re
import numpy as np
import pandas as pd
import nltk
import pickle

nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    input: database file name
    
    outputs:
        X - messages
        Y - anything beside messages
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    #engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table(database_filepath,engine)
    
    #Define feature and target variables X and Y
    X = df["message"]
    Y = df.drop(['id','message','original','genre'], axis=1) #Y = df.iloc[:,4:]
    
    category_names = Y.columns.values
    
    return X, Y, category_names
    


def tokenize(text):
    """
    input: message that needs to be tokenized
    
    output: message that is normalized, lemmatized, with leading/trailing white spaces removed 
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    This builds the model pipeline
    output  - An ML pipeline that applies a classifier and process messages
    """
    
    #pipeline creation
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
         ])
    
    # grid search parameters creation
    parameters = {
        'tfidf__use_idf': (True,False),
        'clf__estimator__n_estimators': [50,100],
        'clf__estimator__learning_rate': [0.5],
        'clf__estimator__random_state': [42]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This evaluates the model and reports on f1 score, precision and recall of each output category of the dataset
    Arguments:
        model - the classification model
        X_test - messages in the test set
        Y_test - messaages in the target test set
        category_names - the category labels of the dataset
    """
    
    y_pred = model.predict(X_test)
    
    #model accuracy
    model_accuracy  = (y_pred == Y_test.values).mean().mean()
    print('The model accuracy is {:.2f}'.format(model_accuracy))
    
    #print classifcation report ie. f1 score, precison and recall for test set
    print(classification_report(Y_test, y_pred, target_names= category_names))
    

def save_model(model,model_filepath):
    """
    This saves the model as a pickle file
    Arguments:
    model - classification model
    model_filepath - path of the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
          
          
def main():
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


if __name__ == '__main__':
    main()