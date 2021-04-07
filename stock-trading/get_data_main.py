import pandas_datareader as pdr
import pandas as pd
import math
from finta import TA
from sklearn.preprocessing import MinMaxScaler

# Prepares our dataframe for training
def prepare_data_for_training(stock, pred, start_time, end_time, short_term=21):
    
    data = get_data(stock, start_time, end_time)  # get the stock historical data
    data = add_indicators(data)  # add technical indicators for each stock
    data = calculate_future_result(data, short_term)  # add expected future result for given set of features
    data = scale_features(data)  # scale close price and volume with minmax
    
    # if data is coming for the model training, remove more first and last data point
    if pred == 0:
        data = data[100:]  # Some indicators need longer time to be avilalbe so we can just remove first 100
        data = data[:-short_term]  # In order to accurately train on future success we need to eliminate not fully completed entries, ie. last days for which we cannot give predictions
    else:
        data = data.tail(1)  # if it is data for predictions, leave just last day
        
    return data

def get_data(stock, start_time, end_time):
    

    data = pdr.DataReader(stock, 'yahoo', start_time, end_time)  # get the data from Yahoo
    
    # Getting the S&P500 relative price difference. A 5% gain is not impressive, if S&P gained 10%
    SP = pdr.DataReader('SPY', 'yahoo', start_time, end_time)  # get s&p data from yahoo
    SP['sp_percent_change'] = SP['Adj Close'].pct_change(periods=1).astype(float)  # calculate daily change in s&p
    data = data.merge(SP['sp_percent_change'], left_index=True, right_index=True)  # add Ss&p data to stock df
    data['percent_change'] = data['Adj Close'].pct_change(periods=1).astype(float)  # add daily change for stock data
    
    # Calculate Daily percent change as compared to the S&P500
    data['relative_change'] = data['percent_change'] - data['sp_percent_change']  # get the difference
    
    data.reset_index(inplace=True)  # reset index
    data["ticker"] = stock  # add the ticker column
    data.columns = [x.lower() for x in data.columns]  # change column names to lowercase
    
    return data

# Adds indicators to dataframe of this given stock
def add_indicators(data):
    
    # This is a list of all possible indicators we can use
    # indicators = ['ADL', 'ADX', 'AO', 'APZ', 'ATR',
    # 'BASP', 'BASPN', 'BBANDS', 'BBWIDTH', 'CCI', 'CFI', 'CHAIKIN', 'CHANDELIER', 'CMO', 'COPP', 'DEMA', 'DMI',
    # 'DO', 'EBBP', 'EFI', 'EMA', 'EMV', 'ER', 'EVWMA', 'EV_MACD', 'FISH', 'FVE', 'HMA', 'ICHIMOKU', 'IFT_RSI',
    # 'KAMA', 'KC', 'KST', 'MACD', 'MFI', 'MI', 'MOM', 'MSD', 'OBV', 'PERCENT_B', 'PIVOT', 'PIVOT_FIB', 'PPO',
    # 'PZO', 'QSTICK', 'ROC', 'RSI', 'SAR', 'SMA', 'SMM', 'SMMA', 'SQZMI', 'SSMA', 'STC', 'STOCH', 'STOCHD',
    # 'STOCHRSI', 'TEMA', 'TMF', 'TP', 'TR', 'TRIMA', 'TRIX', 'TSI', 'UO', 'VAMA', 'VFI', 'VORTEX', 'VPT', 'VR',
    # 'VWAP', 'VW_MACD', 'VZO', 'WILLIAMS', 'WMA', 'WOBV', 'WTO', 'ZLEMA']

    # Here are all indicators we are using
    indicators = ['BASP', 'EMV', 'VFI', 'EBBP', 'MACD', 'ER', 'MI', 'ATR', 'DEMA', 
                  'ADX', 'CMO', 'MFI', 'ROC', 'STOCHRSI', 'TRIX', 'EMA']

    
    # These indicators need more tuning or are broken
    broken_indicators = ['SAR', 'TMF', 'VR', 'QSTICK', 'FRAMA']
    
    # Loop for each indicator
    for indicator in indicators:
        if indicator not in broken_indicators:  # check if it is not "broken"
        
            # Using python's eval function to create a method from a string instead of having every method defined 
            df = None  # set empty df object
            df = eval('TA.' + indicator + '(data)')  # calculate and add technical indicator
        
                
            if not isinstance(df, pd.DataFrame):   # Some method return series, so we can check to convert here
                df = df.to_frame()
            df = df.add_prefix(indicator + '_')  # Appropriate labels on each column
            data = data.merge(df, left_index=True, right_index=True)  # Join merge dataframes based on the date
                
    data.columns = data.columns.str.replace(' ', '_')  # Fix labels
    
    # Make some adjustments to some indicators
    data['DEMA_9_period_DEMA'] = data['DEMA_9_period_DEMA'].pct_change(periods=1)  # if it is DEMA indicator, make it change of DEMA
    data['MACD_MACD'] = data['MACD_MACD'].pct_change(periods=1)  # also change for MACD
    del data['MACD_SIGNAL']  # delete other MACD indicator
    data['ATR_14_period_ATR'] =  data['ATR_14_period_ATR'] / data['adj_close']  # offset ATR with close price
    data['EBBP_Bull.'] = data['EBBP_Bull.'] / data['EMA_9_period_EMA']  # offset EBBP with EMA
    data['EBBP_Bear.'] = data['EBBP_Bull.'] / data['EMA_9_period_EMA']  # offset EBBP with EMA
    del data['EMA_9_period_EMA']  # del EMA as we don't use it

    return data

# Get the result of future stock position (with default interval of 21 days, ie. approx. trading month)
def calculate_future_result(data, short_term=21):
    
    data['short_result'] = None  # add column
    
    # Loop for each row
    for index, row in data.iterrows():
        
        percent_change = data.loc[index + 1:index + short_term]['relative_change'].sum() * 100  # Sums the total relative change compared to S&P over
        
        if math.isnan(percent_change): # if result is nan, make it a zero
            percent_change = 0
        else:  # else just round it to integer
            percent_change = int(round(percent_change))
            
        data.at[index, 'short_result'] = percent_change  # add result to df
        
    del data['relative_change']  # delete relative change variable, as it is not neede after new result was calculated 

    return data

# Scale close price and volume with MinMax scaler
def scale_features(data):
    
    mms = MinMaxScaler()
    data[['adj_close','volume']] = mms.fit_transform(data[['adj_close','volume']])
    
    return data