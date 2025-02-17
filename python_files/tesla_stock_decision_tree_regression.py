# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import numpy as np  # For sorting

# Ensure that the 'plots/tesla_stock_decision_tree_regression' directory exists
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/tesla_stock_decision_tree_regression')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load cleaned data
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Prepare the feature and target variables
# Using 'Close' price as the target and using 'Open', 'High', 'Low', and 'Volume' as features
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Apply a moving average to smooth the predictions
window_size = 10  # Increase window size for smoother predictions
y_pred_smoothed = pd.Series(y_pred).rolling(window=window_size).mean()

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Sort the actual values (y_test) and predictions (y_pred) by date for correct alignment
y_test_sorted = y_test.sort_index()  # Sort the actual values by date
y_pred_sorted = y_pred[np.argsort(y_test_sorted.index)]  # Sort predictions by the same date order
y_pred_smoothed_sorted = y_pred_smoothed[np.argsort(y_test_sorted.index)]  # Sort smoothed predictions

# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))

# Plot actual closing price as a line
plt.plot(y_test_sorted.index, y_test_sorted, label='Actual Closing Price', color='blue', linewidth=2)

# Plot smoothed predicted closing price as a line (instead of scatter)
plt.plot(y_test_sorted.index, y_pred_smoothed_sorted, label='Smoothed Predicted Closing Price', color='red', linestyle='dashed', linewidth=2)

# Title and labels
plt.title('Tesla Stock Price - Actual vs Smoothed Predicted (Decision Tree Regression)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)

# Add legend
plt.legend()

# Save the plot to the specified directory
plt.tight_layout()  # Ensure proper layout with date labels
plt.savefig(os.path.join(plots_dir, 'actual_vs_smoothed_predicted.png'))
plt.show()
