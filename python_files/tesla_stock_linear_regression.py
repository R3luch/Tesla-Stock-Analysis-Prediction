# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Ensure that the 'tesla_stock_linear_regression' directory exists within 'plots'
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/tesla_stock_linear_regression')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the cleaned Tesla stock data
data = pd.read_csv('../data/tesla_stock_data_processed.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Feature engineering: We will use the 'Close' price for the previous day(s) as predictors
# Shift 'Close' by 1 day to predict the next day's closing price
data['Close_Lag1'] = data['Close'].shift(1)

# Drop any rows with NaN values (in case the first row has NaN after shifting)
data = data.dropna()

# Select features (X) and target (y)
X = data[['Close_Lag1']]  # Using the previous day's closing price to predict the next day's
y = data['Close']  # The target is the next day's closing price

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the actual vs predicted closing prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='dodgerblue')
plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='orange', linestyle='--')
plt.title('Tesla Stock Price Prediction - Linear Regression')
plt.xlabel('Date')
plt.ylabel('Price [USD]')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'linear_regression_prediction.png'))  # Save the plot in the correct folder
plt.show()
