# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

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

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='red', linestyle='dashed', linewidth=2)
plt.title('Tesla Stock Price - Actual vs Predicted (Decision Tree Regression)')
plt.xlabel('Date')
plt.ylabel('Price [USD]')
plt.legend()

# Save the plot to the specified directory
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
plt.show()
