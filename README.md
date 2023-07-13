# Multi-stage Continuous-Flow Manufacturing Process Prediction

This project aims to predict certain properties of a multi-stage continuous-flow manufacturing process. The goal is to develop predictive models that can be used in real-time production environments for tasks such as process control and anomaly detection.

## Dataset

The dataset used for this project is derived from an actual production run spanning several hours. It contains data from a continuous-flow manufacturing process with multiple stages. The data is sampled at 1 Hz.

- The first stage consists of Machines 1, 2, and 3 operating in parallel, with their outputs combined by a step.
- Measurements are made in 15 locations to predict primary properties of the output from the combiner.
- The output flows into a second stage with Machines 4 and 5 processing in series.
- Measurements are made again in the same 15 locations to predict secondary properties.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Approach (Added In Update V1.1)

The project involves training and evaluating different machine learning models on the dataset. The models are trained to predict the target properties of the line's output based on the provided input data. The following models were used:

1. Linear Regression
2. Random Forest
3. Lasso Regression
4. Ridge Regression
5. Gradient Boosting
6. AdaBoost
7. Support Vector Regression (SVR)
8. K-Nearest Neighbors (KNN)
9. Multi-Layer Perceptron (MLP)
10. XGBoost

## Code

The code uses Python and scikit-learn library for training and evaluating the models. The dataset is loaded and preprocessed, and then the models are trained and evaluated. The evaluation metrics used are Mean Squared Error (MSE) and Mean Absolute Error (MAE).

```python
# Code snippet for training and evaluating the models
# ...

# Print the results for each model
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model: {type(model).__name__}")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("---------------------")
```

## Usage

1. Install the required dependencies using pip:

```
pip install pandas scikit-learn
```

2. Preprocess the data: Load and preprocess the dataset by handling missing values, outliers, and performing necessary transformations or feature engineering. This step can be done using the appropriate functions or libraries for data cleaning and feature manipulation.

3. Train and evaluate the models: Use the provided Python script or modify it based on your specific requirements. The script assumes that the preprocessed data is loaded into `X` (input features) and `y` (output labels) variables. You can modify the script to incorporate different models or techniques by importing them from scikit-learn and replacing the `LinearRegression` model with your chosen model. Train the model using the training set, make predictions on the test set, and evaluate the model using appropriate evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

4. Model optimization: Experiment with different models, hyperparameters, or techniques to improve the performance of your predictions. You can try techniques like hyperparameter tuning using grid search or random search.

5. Deployment: Once you have selected and optimized the best-performing model, you can deploy it in a real-time production environment for use in process control or anomaly detection.

## Output (Of Base Version)
```
Mean Squared Error: 1.262177448353619e-29
Mean Absolute Error: 3.552713678800501e-15
```

<a href="https://imgur.com/RqkU2Dm.png"><img src="https://imgur.com/RqkU2Dm.png" title="source: imgur.com" /></a>

## Output (Of Update V1.1)

Here are the MSE and MAE observations for each model:

| Model                       | Mean Squared Error     | Mean Absolute Error   |
|-----------------------------|-----------------------|-----------------------|
| Linear Regression           | 1.262177448353619e-29 | 3.552713678800501e-15 |
| Random Forest               | 6.735105082159746e-25 | 8.206768598029157e-13 |
| Lasso                       | 1.262177448353619e-29 | 3.552713678800501e-15 |
| Ridge                       | 1.262177448353619e-29 | 3.552713678800501e-15 |
| Gradient Boosting           | 1.262177448353619e-29 | 3.552713678800501e-15 |
| AdaBoost                    | 7.532122809140743e-24 | 2.744471316873387e-12 |
| SVR (_SVR achieved perfect predictions with 0.0 errors. This result should be interpreted with caution as it could indicate overfitting or other issues._)| 0.0                   | 0.0                   |
| K-Nearest Neighbors (KNN) (_KNeighborsRegressor also achieved perfect predictions with 0.0 errors. Similar to SVR, this result should be carefully analyzed for potential issues._)   | 0.0                   | 0.0                   |
| Multi-Layer Perceptron (MLP) | 27.77796198667492     | 5.224694540054797     |
| XGBoost                     | 8.381903187151819e-13 | 9.155273446026513e-07 |

- Please note that the observations for SVR and K-Nearest Neighbors (KNN) indicate perfect predictions with 0.0 errors. 
- However, it's important to further investigate these results, considering factors such as overfitting, data anomalies, or other potential issues.

<a href="https://imgur.com/RVwhJaW.png"><img src="https://imgur.com/RVwhJaW.png" title="source: imgur.com" /></a>

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

- The dataset used in this project is provied by [UpSkill Campus](https://drive.google.com/file/d/1yvZzslpbWw2mpCVF5QqueSkNrNHmtvDE/view?usp=share_link)

## Conclusion
In this project, we trained and evaluated various machine learning models to predict certain properties of a continuous-flow manufacturing process. The models achieved promising results overall, with some models exhibiting perfect predictions. However, further analysis and validation are required to ensure the robustness and generalization of the models in real-time production environments.
