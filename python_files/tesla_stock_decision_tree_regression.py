# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import numpy as np

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, '../plots/tesla_stock_decision_tree_regression')
data_path = os.path.join(current_dir, '../data/tesla_stock_data_processed.csv')

# Create the directory for saving plots if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Prepare feature variables (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Create a DataFrame with actual and predicted values, sorted by date
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()

# Apply a moving average to smooth the predictions
window_size = 10  # Adjust as needed
results['Predicted_Smoothed'] = results['Predicted'].rolling(window=window_size, min_periods=1).mean()

# Plot actual vs smoothed predicted values
plt.figure(figsize=(14, 7))

# Plot actual closing prices
plt.plot(results.index, results['Actual'], label='Actual Closing Price', color='blue', linewidth=2)

# Plot smoothed predicted prices
plt.plot(results.index, results['Predicted_Smoothed'], label='Smoothed Predicted Closing Price',
         color='red', linestyle='dashed', linewidth=2)

# Set plot title and labels
plt.title('Tesla Stock Price - Actual vs Smoothed Predicted (Decision Tree Regression)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot
plot_path = os.path.join(plots_dir, 'actual_vs_smoothed_predicted.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")





