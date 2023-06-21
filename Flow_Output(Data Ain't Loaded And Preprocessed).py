import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data from a CSV file or any other suitable format
data = pd.read_csv('continuous_factory_process.csv')

# Perform any necessary preprocessing steps, such as handling missing values or outliers,
# and feature engineering. Modify this code based on your specific data and requirements.
# In future updates I will be adding that features also. Till then Enjoy this.

# Extract the input features (X) and output labels (y) from the data
X = data.drop(['output_label'], axis=1)  # Adjust 'output_label' with the actual column name of your output
y = data['output_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
