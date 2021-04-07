"""
Created on Thu Oct 15 17:33:26 2020

@author: afo
"""

import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend
import subprocess
import sys
import boto3

# function to install missing library from inside the script
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

"""
__main__
In order for AWS to train the model when you call the API, you put training in the __main__ block.
"""
if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    install('tables')  # install dependency lib

    # Load data from the location specified by args.train & args.test (In this case, an S3 bucket).
    data = pd.read_hdf(os.path.join(args.train,'train.h5'))
    data_test = pd.read_hdf(os.path.join(args.test,'test.h5'))

    # Delete for prod
    #data = data.sample(1000)

    ## Split to features and classes
    X_train = data.drop(['index', 'short_result'], axis=1, errors='ignore')
    y_train = data['short_result']
    
    X_test = data_test.drop(['index', 'short_result'], axis=1, errors='ignore')
    y_test = data_test['short_result']
    
    ## Naive Bayes parameters & scoring mechanism
    param_grid = [{'n_estimators': [100, 200, 300, 500], 'criterion': ['gini', 'entropy'], 'bootstrap': [True, False]}]
    scores = 'f1_weighted'
    #scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    # Main model object
    with parallel_backend('threading',n_jobs=8):
        model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring=scores, cv=10, n_jobs=-2, 
                             refit = 'f1_weighted')

    # Fit the model and print the best parameters
    model.fit(X_train, y_train.values.ravel())
    print('Best params: %s' % model.best_params_)
    
    #Save the model to the local storage
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Upload model file to S3 bucket
    s3 = boto3.client('s3')
    with open(os.path.join(args.model_dir, "model.joblib"), "rb") as f:
        s3.upload_fileobj(f, args.train.rsplit('/')[2], "model/model.joblib")

    print("Success!")
