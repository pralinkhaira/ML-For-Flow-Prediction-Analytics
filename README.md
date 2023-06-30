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

## Usage

1. Install the required dependencies using pip:

```
pip install pandas scikit-learn
```

2. Preprocess the data: Load and preprocess the dataset by handling missing values, outliers, and performing necessary transformations or feature engineering. This step can be done using the appropriate functions or libraries for data cleaning and feature manipulation.

3. Train and evaluate the models: Use the provided Python script or modify it based on your specific requirements. The script assumes that the preprocessed data is loaded into `X` (input features) and `y` (output labels) variables. You can modify the script to incorporate different models or techniques by importing them from scikit-learn and replacing the `LinearRegression` model with your chosen model. Train the model using the training set, make predictions on the test set, and evaluate the model using appropriate evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

4. Model optimization: Experiment with different models, hyperparameters, or techniques to improve the performance of your predictions. You can try techniques like hyperparameter tuning using grid search or random search.

5. Deployment: Once you have selected and optimized the best-performing model, you can deploy it in a real-time production environment for use in process control or anomaly detection.

## Output
```
Mean Squared Error: 1.262177448353619e-29
Mean Absolute Error: 3.552713678800501e-15
```

<a href="https://imgur.com/RqkU2Dm.png"><img src="https://imgur.com/RqkU2Dm.png" title="source: imgur.com" /></a>


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

- The dataset used in this project is provied by [UpSkill Campus](https://drive.google.com/file/d/1yvZzslpbWw2mpCVF5QqueSkNrNHmtvDE/view?usp=share_link)
