import pandas as pd

# Load the CSV file from the specified path
df = pd.read_csv('../data/tesla_stock_data.csv')

# Display column names to check the structure
print(df.columns)

# Remove rows with missing values and NaNs
df = df.dropna()

# Convert the 'Date' column to the appropriate datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Ensure numeric columns are properly formatted
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# List columns to drop (adjust this list based on actual column names)
columns_to_drop = ['TSLA', 'TSLA.1', 'TSLA.2']

# Drop columns only if they exist in the DataFrame
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Optionally: Remove NaN values after conversion
df = df.dropna()

# Save the cleaned and processed data to a new CSV file
df.to_csv('../data/tesla_stock_data_processed.csv', index=False)
