# tesla_stock_linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

# Define the path to save the trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/linear_regression_model.pkl")

# Load the cleaned Tesla stock data and set 'Date' as the index
data_path = os.path.join(current_dir, "../data/tesla_stock_data_processed.csv")
data = pd.read_csv(data_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Ensure data is sorted by date
data.sort_index(inplace=True)

# Feature engineering: Use the previous day's closing price as a predictor
if 'Close_Lag1' not in data.columns:
    data['Close_Lag1'] = data['Close'].shift(1)

# Drop any rows with NaN values (e.g., the first row after shifting)
data = data.dropna()

# Define features (X) and target (y)
X = data[['Close_Lag1']]  # Predictor: previous day's closing price
y = data['Close']         # Target: current day's closing price

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model for import in another script
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

# Export the trained model as lr_model for use in other scripts
lr_model = model

# Print evaluation metrics
print(f'Model saved at: {model_path}')

# Create the directory for saving plots if it doesn't exist
plots_dir = os.path.join(current_dir, '../plots/tesla_stock_linear_regression')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Predict on the entire dataset
y_pred = model.predict(X)

# Create a DataFrame with actual and predicted values
results = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
results.sort_index(inplace=True)

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(results.index, results['Actual'], label='Actual', color='blue', linewidth=2)
plt.plot(results.index, results['Predicted'], label='Predicted', color='red', linestyle='dashed', linewidth=2)

# Set title and labels
plt.title('Tesla Stock Price - Actual vs Predicted (Linear Regression)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the plot
plot_path = os.path.join(plots_dir, 'actual_vs_predicted_linear_regression.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")
