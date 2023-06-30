# Linear Regression Model Evaluation

## Importing Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

## Reading the Dataset
```python
df = pd.read_csv("D:\\UpSkillCampus Project11\\continuous_factory_process.csv")
```
The code reads the dataset from a CSV file and stores it in a pandas DataFrame called `df`.

## Separating Features and Target
```python
X = df.drop(['time_stamp'], axis=1)
y = df['Stage2.Output.Measurement14.U.Setpoint']
```
The code separates the features and the target variable from the DataFrame. The `X` variable contains the features by dropping the 'time_stamp' column, and the `y` variable contains the target variable 'Stage2.Output.Measurement14.U.Setpoint'.

## Splitting the Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
The code splits the dataset into training and testing sets using the `train_test_split` function. It takes the features (`X`) and target variable (`y`) as input and splits them into `X_train`, `X_test`, `y_train`, and `y_test` sets. The test set size is set to 20% of the total dataset, and a random state of 42 is used for reproducibility.

## Creating and Training the Linear Regression Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
The code creates an instance of the `LinearRegression` model and fits it to the training data using the `fit` method. This step trains the model on the training set.

## Making Predictions on the Test Set
```python
y_pred = model.predict(X_test)
```
The code uses the trained model to make predictions on the test set by calling the `predict` method on the model with the test features (`X_test`) as input. The predicted values are stored in the `y_pred` variable.

## Model Evaluation
```python
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
```
The code calculates the mean squared error (MSE) and mean absolute error (MAE) between the predicted values (`y_pred`) and the actual target values (`y_test`) using the `mean_squared_error` and `mean_absolute_error` functions, respectively. The computed error values are then printed to the console.

This code reads a dataset, splits it into training and testing sets, trains a linear regression model on the training set, and evaluates the model's performance by calculating the MSE and MAE on the test set.
