from stock_list_gathering_hf import get_stocks_predict, get_data_main
import os
from os.path import abspath
from inspect import getsourcefile
import datetime

# Get the working path of script
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

stock_training = get_stocks_predict() # get list of stocks gathered from txt file for prediction
#stock_training =["MSFT", "AAPL", "AMZN", "GOOG", "FB", "JNJ", "WMT", "V", "PG", "JPM", "UNH", "MA", "INTC", "VZ"] # Tests our model with the top 50 S&P Stocks

start_time = '2000-01-01'  # define the start date for start of 21st century
end_time = datetime.datetime.today() - datetime.timedelta(days=1)   # end time is today-1 (ie. the last full day)

pred = 1  # shows if the data is train (0) or for predictions (1)

short_term = 21  # how many day in future to calculate results (forecast)

# Run main data gathering function
get_data_main(p, pred, stock_training, start_time, end_time, short_term)
    
    
