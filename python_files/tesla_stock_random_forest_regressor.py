# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Ensure that the 'plots/tesla_stock_random_forest_regression' directory exists
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/tesla_stock_random_forest_regression')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load cleaned data
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])

# Check for any missing or duplicate data
print(data.isnull().sum())  # Check for missing values
print(data.duplicated().sum())  # Check for duplicate rows

# Sort data by date (if necessary)
data.sort_values('Date', inplace=True)

# Set Date as index
data.set_index('Date', inplace=True)

# Prepare the feature and target variables
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))

# Plot actual and predicted values
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='blue', linewidth=1.5, marker='o', markersize=4)
plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='red', linestyle='dashed', linewidth=1.5, marker='x', markersize=4)

# Title and labels
plt.title('Tesla Stock Price - Actual vs Predicted (Random Forest Regression)')
plt.xlabel('Date')
plt.ylabel('Price [USD]')

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Add a grid for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Save the plot to the specified directory
plt.tight_layout()  # Ensures proper layout with date labels
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_random_forest.png'))
plt.show()
