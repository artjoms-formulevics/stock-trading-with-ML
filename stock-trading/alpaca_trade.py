#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:01:48 2020

@author: afo
"""


import alpaca_trade_api as tradeapi
import pandas as pd
import time, datetime
from os.path import abspath
from inspect import getsourcefile
import os

# Get the working path of script
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

import config

def get_direction(x):
    return {
        0: 'sell',
        1: 'sell',
        2: 'hold',
        3: 'buy',
        4: 'buy'
    }[x]

def get_size(x):
    return {
        0: 2,
        1: 1,
        2: 0,
        3: 1,
        4: 2
    }[x]

#authentication and connection details
api_key = config.alpaca_api_key
api_secret = config.alpaca_api_secret
base_url = 'https://paper-api.alpaca.markets'

# Read saved predictions from csv
pred = pd.read_csv('todays_predictions.csv')

#instantiate REST API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

#obtain account information
account = api.get_account()
# print(account)

for i in range(0, len(pred)):

    symbol = pred.iloc[i,0]
    qty = get_size(pred.iloc[i,1])
    side = get_direction(pred.iloc[i,1])

    if pred.iloc[i,1] != 2:
        
        try:
            # send order
            api.submit_order(symbol=symbol, 
            		qty=qty, 
            		side=side, 
            		time_in_force='gtc', 
            		type='market',
            		client_order_id=str(int(time.mktime(datetime.datetime.now().timetuple())))) # order id need to be unique, set timestamp
            print("Order for: "+ symbol + " placed successfully! Quantity: " + str(qty) + ". Position: " + side + ".")
        except Exception:
            print("Order for: " + symbol + " was not placed!")
    else:
        print("Prediction for: " + symbol + " is " + side + "! No order opened.")
    time.sleep(1)