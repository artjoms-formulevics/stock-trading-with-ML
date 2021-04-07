import bs4 as bs
import requests
import os
from get_data_main import prepare_data_for_training

# Function to run main loop for collecting stocks
def get_data_main(p, pred, stock_training, start_time, end_time, short_term):
    
    # Aggregates data into a list of dataframes
    data_frames=[]
    
    if pred == 0:
        data_file = '/data.csv'
    else:
        data_file = '/data_predict.csv'
    
    # Remove previous  file
    try:
        os.remove(p + data_file)
    except OSError:
        pass
    
    # Main loop through each stock ticker
    for i in range(0, len(stock_training)):
        print(stock_training[i], "Gathering")
        try:  # collect each ticker
            data_frames.append(prepare_data_for_training(stock_training[i], pred, start_time, end_time, short_term))
            if i == 0:  # if it first ticker, add header to csv 
                data_frames[0].to_csv(p + data_file, mode='a', header=True)
            else:
                data_frames[0].to_csv(p + data_file, mode='a', header=False)  # otherwise just write
            print(stock_training[i], "Success")
            data_frames=[]
        # Error in one of stocks
        except Exception as e:
            print(stock_training[i], e)
    

# Function to scare list of S&P 500 stocks from wiki for training data
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')  # specify url
    soup = bs.BeautifulSoup(resp.text, 'lxml')  # create souop object
    table = soup.find('table', {'class': 'wikitable sortable'})  # find a table with data
    
    # Wead all tickers to the list
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())

    # Write tickers in txt file rowwise
    with open('S&P500.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % ticker for ticker in tickers)

# Function to load stock list from txt by rows for training data
def get_stocks(): 
    tickers = []
    f = open("S&P500.txt", "r")
    for x in f:
        x=x.strip()
        if x.isalpha():
            tickers.append(x)
            
    print("\nS&P stock tickers written to file S&P500.txt\n")
            
    return tickers

# Function to load stock list from txt by rows from prediction data
def get_stocks_predict():
    tickers = []
    f = open("stocks_to_predict.txt", "r")
    for x in f:
        x=x.strip()
        if x.isalpha():
            tickers.append(x)
    return tickers