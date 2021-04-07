from stock_list_gathering_hf import save_sp500_tickers, get_stocks, get_data_main
import os
from os.path import abspath
from inspect import getsourcefile
import datetime

# Get the working path of script
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

save_sp500_tickers()  # scrape list of S&P 500 companies from wiki and save to txt
stock_training = get_stocks()  # get list of stocks gathered from txt file
# stock_training =["MSFT", "AAPL", "AMZN", "GOOG", "FB", "JNJ", "WMT", "V", "PG", "JPM", "UNH", "MA", "INTC", "VZ"] # Tests our model with the top 50 S&P Stocks

start_time = '2000-01-01'  # define the start date for start of 21st century
end_time = datetime.datetime.today() - datetime.timedelta(days=1)   # end time is today-1 (ie. the last full day)

pred = 0  # shows if the data is train (0) or for predictions (1)

short_term = 21  # how many day in future to calculate results (forecast)

# Run main data gathering function
get_data_main(p, pred, stock_training, start_time, end_time, short_term)
        