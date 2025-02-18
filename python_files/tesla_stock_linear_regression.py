# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Define paths for the plots directory and the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, '../plots/tesla_stock_linear_regression')
data_path = os.path.join(current_dir, '../data/tesla_stock_data_processed.csv')

# Create the directory for saving plots if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the cleaned Tesla stock data and set 'Date' as the index
data = pd.read_csv(data_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Ensure data is sorted by date
data.sort_index(inplace=True)

# Feature engineering: Use the previous day's closing price as a predictor
data['Close_Lag1'] = data['Close'].shift(1)

# Drop any rows with NaN values (e.g., the first row after shifting)
data = data.dropna()

# Define features (X) and target (y)
X = data[['Close_Lag1']]  # Predictor: previous day's closing price
y = data['Close']         # Target: current day's closing price

# Split the data into training and testing sets (80% training, 20% testing)
# We use shuffle=False to maintain time series order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R-squared metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Plot the actual vs predicted closing prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='dodgerblue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='orange', linestyle='--', linewidth=2)
plt.title('Tesla Stock Price Prediction - Linear Regression', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot in the specified directory
plot_path = os.path.join(plots_dir, 'linear_regression_prediction.png')
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")
