#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:48:17 2020

@author: afo
"""

import pandas as pd
from inspect import getsourcefile
from os.path import abspath
import os
import joblib
import boto3

import config

# Get the working path of script
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

# Load predict dataset
X_test = pd.read_csv(p + '/data_predict.csv', index_col=0)

model_prefix = 'rf_model'  # change for the model name based on algorithm

# Load model
fname = 'model/model.joblib' # file name & location on S3
local_name = 'model.joblib'  # name to save to

# Connect to S3 & download model, encoder & scaler
s3 = boto3.client('s3')
s3.download_file(config.bucket_name, fname, p + '/rf_model/' + local_name)
s3.download_file(config.bucket_name, 'pickles/scaler.gz', p + '/'  +model_prefix + '/scaler.gz')

## Load downlaoded data
mdl = joblib.load(p + '/' + model_prefix + '/model.joblib')  # load model
scaler = joblib.load(p + '/' + model_prefix + '/scaler.gz')  # load scaler

d = str(X_test.iloc[0,0])  # get the date for which predictions are made

# remove the extra columns
X_test = X_test.drop(['deltawma', 'index', 'short_result', 'date'], axis =1, errors='ignore')
X_test = X_test.drop(['high', 'low', 'open', 'close'], axis =1, errors='ignore')
X_test = X_test.drop(['up_move', 'down_move', 'plus', 'minus'], axis =1, errors='ignore')  # delete extra columns

X_test = X_test.dropna()  # drop NAs if any

# Remove ticker name to store it separetely
tickers = X_test['ticker']  
X_test = X_test.drop(['ticker'], axis =1, errors='ignore')

X_test = scaler.transform(X_test)  #scale by predefined scaler

# Make predictions
# means buy=0, hold=1. sell=2, strong buy=3, strong sell=4
y_pred = mdl.predict(X_test)
pred = pd.DataFrame({'ticker':tickers,'predict':y_pred})

# Show the output
print('Prediction for %s:' % d)
print()
print(pred.to_string(index=False))
print()
print('Means strong sell=0, sell=1. hold=2, buy=3, strong buy=4')

# Save predictions as csv to use for trading
pred.to_csv('todays_predictions.csv', index=False)
