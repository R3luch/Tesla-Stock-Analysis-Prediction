##### 2025-02 Tesla Stock Analysis & Prediction
##### Import necessary libraries
import yfinance as yf
import pandas as pd

def download_tesla_data(start_date='2015-01-01', end_date='2023-01-01'):
    ticker = 'TSLA'
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

if __name__ == '__main__':
    df = download_tesla_data()
    print(df.head())
    # Save data to CSV file in data/
    df.to_csv('../data/tesla_stock_data.csv', index=False)