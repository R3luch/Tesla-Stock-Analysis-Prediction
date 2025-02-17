# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Ensure that the 'plots/tesla_stock_model_comparison' directory exists
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/tesla_stock_model_comparison')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load cleaned data
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Prepare the feature and target variables
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Calculate Mean Squared Error for all models
dt_mse = mean_squared_error(y_test, dt_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

# Calculate R-squared for all models
dt_r2 = r2_score(y_test, dt_pred)
rf_r2 = r2_score(y_test, rf_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Calculate Mean Absolute Error for all models
dt_mae = mean_absolute_error(y_test, dt_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

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

# Plot the actual vs predicted values for all models
plt.figure(figsize=(14, 7))

# Plot for Decision Tree
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='blue', linewidth=1.5, marker='o', markersize=4)
plt.plot(y_test.index, dt_pred, label='Decision Tree Predicted', color='red', linestyle='dashed', linewidth=1.5, marker='x', markersize=4)

# Plot for Random Forest
plt.plot(y_test.index, rf_pred, label='Random Forest Predicted', color='green', linestyle='dotted', linewidth=1.5, marker='s', markersize=4)

# Plot for Linear Regression
plt.plot(y_test.index, lr_pred, label='Linear Regression Predicted', color='orange', linestyle='-.', linewidth=1.5, marker='^', markersize=4)

# Title and labels
plt.title('Tesla Stock Price - Actual vs Predicted (Comparison of Models)')
plt.xlabel('Date')
plt.ylabel('Price [USD]')

# Add model comparison text to the plot
plt.text(0.02, 0.95, f"Decision Tree MSE: {dt_mse:.4f}\nR²: {dt_r2:.4f}\nMAE: {dt_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='red')
plt.text(0.02, 0.85, f"Random Forest MSE: {rf_mse:.4f}\nR²: {rf_r2:.4f}\nMAE: {rf_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='green')
plt.text(0.02, 0.75, f"Linear Regression MSE: {lr_mse:.4f}\nR²: {lr_r2:.4f}\nMAE: {lr_mae:.4f}", transform=plt.gca().transAxes, fontsize=10, color='orange')

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Add a grid for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Save the plot to the specified directory
plt.tight_layout()  # Ensures proper layout with date labels
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_comparison_with_metrics.png'))
plt.show()
