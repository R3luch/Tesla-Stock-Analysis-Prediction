# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure that the 'plots' directory exists
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load data from the processed CSV file
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Ensure Date is sorted (in case it's not)
data = data.sort_index()

# Display basic information about the data
print(data.head())
print(data.describe())

# Plot the closing price of Tesla stock
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Closing Price', color='dodgerblue', linewidth=2)
plt.title('Tesla Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price [USD]')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'closing_price.png'))  # Save the plot as a PNG file
plt.show()

# Calculate moving averages: 50-day and 200-day
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Plot closing price with moving averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Closing Price')
plt.plot(data['SMA_50'], label='50-day SMA')
plt.plot(data['SMA_200'], label='200-day SMA')
plt.title('Tesla Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price [USD]')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'price_with_moving_averages.png'))  # Save the plot as a PNG file
plt.show()
