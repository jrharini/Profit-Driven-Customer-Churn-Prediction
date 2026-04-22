Profit-Driven-Customer-Churn-Prediction

OVERVIEW:-
This project builds a machine learning pipeline to predict customer churn with a focus on maximizing business profit rather than just improving model accuracy. It combines predictive modeling with profit-maximizing decision rules and uses Golden Eagle Optimization for feature selection and decision-threshold tuning.
Problem Statement
Traditional churn prediction models are built to maximize accuracy, but high accuracy does not always mean high profit. This project addresses that gap by shifting the focus from accuracy-based evaluation to profit-driven decision making.

TECH STACK:-
Python
Pandas & NumPy — data preprocessing and manipulation
Scikit-learn — model building and evaluation
Streamlit — interactive web app for predictions
Matplotlib / Seaborn — data visualization

APPROACH:-
Data Preprocessing — cleaned and prepared the customer dataset for modeling
Feature Selection — applied Golden Eagle Optimization (GEO) to select the most relevant features
Model Training — used Logistic Regression as the base classifier
Probability Calibration — validated prediction reliability using Brier Score analysis
Decision Threshold Tuning — used GEO to find the optimal threshold that maximizes profit
Profit Evaluation — compared profit outcomes against baseline accuracy-based models

RESULTS:-
Logistic Regression chosen as base classifier based on Brier Score validation
Golden Eagle Optimization successfully reduced feature space and tuned decision threshold
Profit-driven pipeline outperformed traditional accuracy-based models in financial impact
