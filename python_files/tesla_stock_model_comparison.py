# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib
import numpy as np

# Define paths for the plots directory and the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, '../plots/tesla_stock_model_comparison')
data_path = os.path.join(current_dir, '../data/tesla_stock_data_processed.csv')

# Create the directory for saving plots if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv(data_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Feature engineering: Use the previous day's closing price as a predictor
if 'Close_Lag1' not in data.columns:
    data['Close_Lag1'] = data['Close'].shift(1)

# Drop any rows with NaN values (e.g., the first row after shifting)
data = data.dropna()

# Prepare feature variables (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Volume']]  # Used for DT and RF models
y = data['Close']

# For the Linear Regression model, only use the previous day's closing price (Close_Lag1)
X_lr = data[['Close_Lag1']]  # Features for Linear Regression

# Load the models from the saved .pkl files in the ../models/ folder
dt_model = joblib.load(os.path.join(current_dir, '../models/decision_tree_model.pkl'))
rf_model = joblib.load(os.path.join(current_dir, '../models/random_forest_model.pkl'))
lr_model = joblib.load(os.path.join(current_dir, '../models/linear_regression_model.pkl'))

# Make predictions using the loaded models
dt_pred = dt_model.predict(X)
rf_pred = rf_model.predict(X)
lr_pred = lr_model.predict(X_lr)  # Use X_lr for Linear Regression

# Create a DataFrame with actual and predicted values
results = pd.DataFrame({
    'Actual': y,
    'DecisionTree': dt_pred,
    'RandomForest': rf_pred,
    'LinearRegression': lr_pred
})

# Calculate evaluation metrics for each model
dt_mse = mean_squared_error(results['Actual'], results['DecisionTree'])
rf_mse = mean_squared_error(results['Actual'], results['RandomForest'])
lr_mse = mean_squared_error(results['Actual'], results['LinearRegression'])

dt_r2 = r2_score(results['Actual'], results['DecisionTree'])
rf_r2 = r2_score(results['Actual'], results['RandomForest'])
lr_r2 = r2_score(results['Actual'], results['LinearRegression'])

dt_mae = mean_absolute_error(results['Actual'], results['DecisionTree'])
rf_mae = mean_absolute_error(results['Actual'], results['RandomForest'])
lr_mae = mean_absolute_error(results['Actual'], results['LinearRegression'])

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

dt_mape = calculate_mape(results['Actual'], results['DecisionTree'])
rf_mape = calculate_mape(results['Actual'], results['RandomForest'])
lr_mape = calculate_mape(results['Actual'], results['LinearRegression'])

# Calculate Adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n = len(results)  # Number of data points
p_dt = X.shape[1]  # Number of features for Decision Tree and Random Forest
p_lr = X_lr.shape[1]  # Number of features for Linear Regression

dt_adj_r2 = adjusted_r2(dt_r2, n, p_dt)
rf_adj_r2 = adjusted_r2(rf_r2, n, p_dt)
lr_adj_r2 = adjusted_r2(lr_r2, n, p_lr)

# Print the comparison results
print(f"Decision Tree Regressor MSE: {dt_mse:.4f}")
print(f"Random Forest Regressor MSE: {rf_mse:.4f}")
print(f"Linear Regression MSE: {lr_mse:.4f}")
print(f"Decision Tree Regressor R²: {dt_r2:.4f}")
print(f"Random Forest Regressor R²: {rf_r2:.4f}")
print(f"Linear Regression R²: {lr_r2:.4f}")
print(f"Decision Tree Regressor MAE: {dt_mae:.4f}")
print(f"Random Forest Regressor MAE: {rf_mae:.4f}")
print(f"Linear Regression MAE: {lr_mae:.4f}")
print(f"Decision Tree Regressor MAPE: {dt_mape:.4f}%")
print(f"Random Forest Regressor MAPE: {rf_mape:.4f}%")
print(f"Linear Regression MAPE: {lr_mape:.4f}%")
print(f"Decision Tree Regressor Adjusted R²: {dt_adj_r2:.4f}")
print(f"Random Forest Regressor Adjusted R²: {rf_adj_r2:.4f}")
print(f"Linear Regression Adjusted R²: {lr_adj_r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.scatter(results.index, results['Actual'], label='Actual', color='blue', s=20, alpha=0.6, marker='o')
plt.scatter(results.index, results['DecisionTree'], label='Decision Tree', color='red', s=20, alpha=0.6, marker='x')
plt.scatter(results.index, results['RandomForest'], label='Random Forest', color='green', s=20, alpha=0.6, marker='s')
plt.scatter(results.index, results['LinearRegression'], label='Linear Regression', color='orange', s=20, alpha=0.6, marker='^')

# Labels and title
plt.title('Tesla Stock Price - Actual vs Predicted', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)

# Add model comparison metrics as text on the plot (top-left corner)
textstr = '\n'.join((
    f'Decision Tree R²: {dt_r2:.4f}',
    f'Random Forest R²: {rf_r2:.4f}',
    f'Linear Regression R²: {lr_r2:.4f}',
    f'Decision Tree MSE: {dt_mse:.4f}',
    f'Random Forest MSE: {rf_mse:.4f}',
    f'Linear Regression MSE: {lr_mse:.4f}',
    f'Decision Tree MAE: {dt_mae:.4f}',
    f'Random Forest MAE: {rf_mae:.4f}',
    f'Linear Regression MAE: {lr_mae:.4f}',
    f'Decision Tree MAPE: {dt_mape:.4f}%',
    f'Random Forest MAPE: {rf_mape:.4f}%',
    f'Linear Regression MAPE: {lr_mape:.4f}%',
    f'Decision Tree Adjusted R²: {dt_adj_r2:.4f}',
    f'Random Forest Adjusted R²: {rf_adj_r2:.4f}',
    f'Linear Regression Adjusted R²: {lr_adj_r2:.4f}'
))

# Position of the text on the plot (left top corner)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.7))

# Adjust the position of the legend (right side of the plot)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot
plot_path = os.path.join(plots_dir, 'actual_vs_predicted_scatter_comparison_with_additional_metrics.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")
