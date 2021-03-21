# Regression task description
This repository contains analytics of warehouse and retail sales dataset. 

This repo has following directory structure:
```
.
├───data
│   ├───data_regression_for_task.csv - warehouse and retail sales dataset
├───cross_validation.py - auxiliary file for performing temporary cross-validation
├───data_analysis.ipynb - data analytics and visualization of data dependencies
├───data_preprocessing.ipynb - preparing data for a regression model
|───processing.py - auxiliary file for performing data preprocessing
└───regression.ipynb - training of regression models and evaluation of a forecast for the last month for "SALES"
```

The quality of regression models is evaluated using the function:

<img src="https://render.githubusercontent.com/render/math?math=\Large\text{Metric}\left(y_{true}, y_{pred}\right) = 100*\frac{\sum\big|y_{true} - y_{pred}\big|}{\sum y_{pred}}">

```python

def metric(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/np.sum(y_pred)*100
```
