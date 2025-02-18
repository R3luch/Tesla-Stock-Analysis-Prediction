# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import joblib  # For saving the trained model
import numpy as np  # For smoothing operations

# Ensure that the 'plots/tesla_stock_random_forest_regressor' directory exists
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/tesla_stock_random_forest_regressor')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define the path for the data file
data_path = os.path.join('../data/tesla_stock_data_processed.csv')

# Create the directory for saving plots if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the cleaned data
data = pd.read_csv(data_path, parse_dates=['Date'])

# Print missing values and duplicate rows
print(data.isnull().sum())      # Check for missing values
print(data.duplicated().sum())  # Check for duplicate rows

# Sort data by date and set 'Date' as the index
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Prepare the feature variables (X) and the target variable (y)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model for import in another script
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/random_forest_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Create a DataFrame with actual and predicted values, sorted by date
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()

# Apply a moving average (rolling mean) to smooth the predictions
window_size = 10  # Adjust window size for smoother predictions
results['Predicted_Smoothed'] = results['Predicted'].rolling(window=window_size, min_periods=1).mean()

# Plot the actual vs smoothed predicted values
plt.figure(figsize=(14, 7))

# Plot actual closing prices
plt.plot(results.index, results['Actual'], label='Actual Closing Price', color='blue', linewidth=2)

# Plot smoothed predicted prices
plt.plot(results.index, results['Predicted_Smoothed'], label='Smoothed Predicted Closing Price',
         color='red', linestyle='dashed', linewidth=2)

# Set title and labels
plt.title('Tesla Stock Price - Actual vs Smoothed Predicted (Random Forest Regression)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Add a grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the plot to the specified directory
plot_path = os.path.join(plots_dir, 'actual_vs_smoothed_predicted_random_forest.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")
print(f"Model saved at: {model_path}")
