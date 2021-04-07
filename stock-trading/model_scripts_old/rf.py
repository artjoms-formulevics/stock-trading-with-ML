#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from inspect import getsourcefile
from os.path import abspath
import os
import joblib
import boto3

import config

p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 2)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

model_prefix = 'rf'

# Load model
train_path = 'data/train/train.h5' # file name & location on S3
test_path = 'data/test/test.h5'  # name to save to
val_path = 'data/val/val.h5'

# Connect to S3 & download model, encoder & scaler
s3 = boto3.client('s3')
s3.download_file(config.bucket_name, train_path, p + '/train.h5')
s3.download_file(config.bucket_name, test_path, p + '/test.h5')
s3.download_file(config.bucket_name, val_path, p + '/val.h5')

# Load data from the location specified by args.train & args.test (In this case, an S3 bucket).
train = pd.read_hdf(p + '/train.h5')
test = pd.read_hdf(p + '/test.h5')
val = pd.read_hdf(p + '/val.h5')

## Split to features and classes
X_train = train.drop(['index', 'short_result'], axis=1, errors='ignore')
y_train = train['short_result']

X_test = test.drop(['index', 'short_result'], axis=1, errors='ignore')
y_test = test['short_result']

X_val = val.drop(['index', 'short_result'], axis=1, errors='ignore')
y_val = val['short_result']

## Random Forest Classifer

param_grid = [{'n_estimators': [100, 200, 300, 500], 'criterion': ['gini', 'entropy'], 'bootstrap': [True, False]}]
scores = 'f1_weighted'
#scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

clf = GridSearchCV(RandomForestClassifier(verbose=10), param_grid=param_grid, scoring=scores, cv=5, n_jobs=-2, 
                   refit = 'f1_weighted', verbose=10)

clf.fit(X_train, y_train.ravel())
print('Best params: %s' % clf.best_params_)

## Getting results overview
y_true, y_pred = y_test, clf.predict(X_test)

#Save the model to the local storage
joblib.dump(clf, p+"/model_" + model_prefix +".joblib")

# Upload model file to S3 bucket
s3 = boto3.client('s3')
with open(p + "/model_" + model_prefix + ".joblib", "rb") as f:
    s3.upload_fileobj(f, config.bucket_name, "model/model.joblib")

## End

print("All Done!")

