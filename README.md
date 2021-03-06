# Disaster-Responses-Pipeline

![image1](https://github.com/blessokeke/Disaster-Responses-Pipeline/blob/main/images/disaster_response.png)

## Project Overview
This project focuses on analyzing disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster message. The dataset contains real messages that were sent during a disaster e.g earthquake or tsunami. 

In this project, I will be creating a machine learning pipeline to categorize these events so that the messages would be sent to the appropriate disaster relief agency. A web app would also be built to enable emergency workers input new messages and get the classification results in several categories and display visualizations of the data.

## File Description
This project is divided into 3 components.
1. **ETL Pipeline:** This loads the messages and categories datasets, merges both datasets, cleans them and stores it in a SQLite database. 
    - The `disaster_messages.csv` and `disaster_categories.csv` contains the disaster messages and categories respectively.
    - The python script `process_data.py` contains the code.
    - `DisasterResponse.db` is the output generated from process_data.py
3. **ML Pipeline:** This part loads the data from the SQLite database, splits the dataset into training and test sets. It also builds a text processing and machine learning pipeline, trains and tunes the model using GridSearchCV. It then outputs the results on the test set and exports the final model as a pickle file. 
    - The python script `train_classifier.py` contains the code.
    - `classifier.pkl` is the output pickle fine generated from train_classifier.py
5. **Flask Web App:** This contains the python script `run.py` to initiate the web app.

## How to run the python scripts
To run the python scripts, follow the instructions below:
1. For the ETL pipeline, run the **process_data.py:** script. In your terminal, run the command `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. For the ML pipeline, run the **train_classifier.py** script. Run `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. For the web app, you could run it from your local machine or from the Udacity workspace.
#### For Local Machine
1. Run the command `python run.py`
2. Go to `http://localhost:3001` to view the app.
#### For Udacity Workspace 
1. Go to the app's directory and run `python run.py`
2. Open another terminal window and type `env|grep WORK`. The output would look like so `viewxxxxxxxx`
3. Open a new web brower window and type `https://SPACEID-3001.SPACEDOMAIN`. Insert your SPACEID and SPACEDOMAIN from step 4 above to view the app.


### Acknowledgment
Many thanks to Udacity for giving me an opportunity to try out real world problems. Thanks to the udacity mentors for their support as well as Data Scientists who have inspired and provided insights to me through Github and StackOverflow.

This project was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program.

### Results
Below are some visuals from my web app.
![image2](https://github.com/blessokeke/Disaster-Responses-Pipeline/blob/main/images/message_category_count.png)
![image3](https://github.com/blessokeke/Disaster-Responses-Pipeline/blob/main/images/message_category_distribution.png)
![image3](https://github.com/blessokeke/Disaster-Responses-Pipeline/blob/main/images/message_genre.png)

