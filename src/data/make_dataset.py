import yfinance as yf


#todos: pratyush either download or load from somewhere
def download_crypto_data(crypto_name, start_date, end_date):
    """
    Download crypto data from Yahoo Finance
    :param crypto_name: (str) Name of the crypto
    :param start_date: (str) Start date of the data
    :param end_date: (str) End date of the data
    :return: (dataframe) Dataframe of the crypto data
    """
    pass


def download_stock_data(stock_name,start_date,end_date):
    """
    Download stock data from Yahoo Finance
    :param stock_name: (str) Name of the stock
    :param start_date: (str) Start date of the data
    :param end_date: (str) End date of the data
    :return: (dataframe) Dataframe of the stock data
    """
    data = yf.download(stock_name, start=start_date, end=end_date)
    data = data.reset_index()
    data.columns = ['date', 'open', 'high', 'low', 'close', 'adjcp', 'volume']

    return data

def calc_macd(data):
    """
    Calculate MACD from the dataframe
    :param data: (dataframe) Dataframe containing the stock data
    :return: (dataframe) Dataframe containing MACD
    """
    return data    
    

#todos: pratyush
def calculate_indicators(data_df, indicator_list):
    pass

