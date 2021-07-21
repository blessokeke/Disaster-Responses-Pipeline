import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Inputs: message and categories dataset filepath
    
    Arguments: 
        message_filepath: File path to csv file containing messages
        categories_filepath: File path to csv file containing categories
        
    Outputs: returns a dataframe with messages and categories merged together
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages,categories, on='id')
    
    return df
    

def clean_data(df):
    """
    Inputs: dataframe that contain messages and categories
    
    Outputs: returns a dataframe with cleaned messages and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist() # use this row to extract a list of new column names for categories.
    categories.columns = category_colnames # rename the columns of `categories`
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:] #categories[column].transform(lambda x: x[-1:]) 
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop(columns ='categories',axis=1, inplace=True) # drop the original categories column from `df`
    df = pd.concat([df,categories],axis=1,join='inner') # concatenate the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace=True)  # drop duplicates
    df.related.replace(2,1,inplace=True) # There are values with 2 and we need to replace them with value 1
    
    return df


def save_data(df, database_filename):
    """
    Input: 
        df - dataframe with cleaned data 
        database_filename - file name for database
    """
    #database_filename = 'ble_etl_pipeline.db'
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False,if_exists='replace') 


def main():
    """
    This function would do the following
    1. load the message and categories data
    2. Clean the data
    3. Save the data into SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()