# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Define paths for the plots directory and the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, '../plots/tesla_stock_model_comparison')
data_path = os.path.join(current_dir, '../data/tesla_stock_data_processed.csv')

# Create the directory for saving plots if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the cleaned data and set 'Date' as index
data = pd.read_csv(data_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Prepare feature variables (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions for all models
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Create a DataFrame with actual and predicted values
results = pd.DataFrame({
    'Actual': y_test,
    'DecisionTree': dt_pred,
    'RandomForest': rf_pred,
    'LinearRegression': lr_pred
})

# Sort the results by date for correct alignment in the scatter plot
results.sort_index(inplace=True)

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

# Plotting scatter plot for actual vs predicted values from each model
plt.figure(figsize=(14, 7))

# Scatter plot for actual values (using date index for x-axis)
plt.scatter(results.index, results['Actual'], label='Actual', color='blue', s=20, alpha=0.6, marker='o')

# Scatter plots for each model's predictions
plt.scatter(results.index, results['DecisionTree'], label='Decision Tree Predicted', color='red', s=20, alpha=0.6, marker='x')
plt.scatter(results.index, results['RandomForest'], label='Random Forest Predicted', color='green', s=20, alpha=0.6, marker='s')
plt.scatter(results.index, results['LinearRegression'], label='Linear Regression Predicted', color='orange', s=20, alpha=0.6, marker='^')

# Set title and labels
plt.title('Tesla Stock Price - Actual vs Predicted (Model Comparison)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price [USD]', fontsize=12)

# Add model evaluation metrics text on the plot
plt.text(0.02, 0.90, f"Decision Tree\nMSE: {dt_mse:.4f}\nR²: {dt_r2:.4f}\nMAE: {dt_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='red')
plt.text(0.02, 0.75, f"Random Forest\nMSE: {rf_mse:.4f}\nR²: {rf_r2:.4f}\nMAE: {rf_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='green')
plt.text(0.02, 0.60, f"Linear Regression\nMSE: {lr_mse:.4f}\nR²: {lr_r2:.4f}\nMAE: {lr_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='orange')

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Add grid and legend for clarity
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the plot to the specified directory
plot_path = os.path.join(plots_dir, 'actual_vs_predicted_scatter_comparison.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as: {plot_path}")
