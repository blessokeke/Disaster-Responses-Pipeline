# Disaster-Responses-Pipeline
## Project Overview
This project focuses on analyzing disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster message. The dataset contains real messages that were sent during a disaster e.g earthquake or tsunami. 

In this project, I will be creating a machine learning pipeline to categorize these events so that the messages would be sent to the appropriate disaster relief agency. A web app would also be built to enable emergency workers input new messages and get the classification results in several categories and display visualizations of the data.

## File Description
This project is divided into 3 components.
1. ETL Pipeline: This loads the messages and categories datasets, merges both datasets, cleans them and stores it in a SQLite database. The python script `process_data.py` contains the code.
2. ML Pipeline: This part loads the data from the SQLite database, splits the dataset into training and test sets. It also builds a text processing and machine learning pipeline, trains and tunes the model using GridSearchCV, outputs the results on the test set and exports the final model as a pickle file. The python script `train_classifier.py` contains the code.
3. Flask Web App: This contains the python script `run.py` to initiate the web app.

## How to run the python scripts


