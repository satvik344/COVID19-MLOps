# Viva Cheat Sheet — COVID-19 ML + MLOps

## Core ML project

### 1. What is the problem?
It is a regression problem because the target, confirmed COVID-19 cases, is numerical.

### 2. What is the target?
`Confirmed`.

### 3. What is the feature?
`Previous_Day_Confirmed`.

### 4. Why use previous-day cases?
They are available before the prediction day and provide a simple time-dependent feature.

### 5. Why is `shuffle=False`?
The observations are chronological. Training uses earlier observations and testing uses later observations.

### 6. What models are used?
Linear Regression, Decision Tree Regression and Random Forest Regression.

### 7. What metrics are used?
MAE, RMSE and R².

### 8. How is the best model selected?
The baseline compares RMSE and selects the lowest RMSE.

---

# Lab 3 — Baseline Pipeline

### 9. What is a pipeline?
A repeatable sequence of preprocessing, training and evaluation steps.

### 10. What does Lab 3 run?
`preprocess.py -> train.py -> evaluate.py`.

### 11. What is saved?
Processed arrays, metadata, model files, predictions and comparison reports.

---

# Lab 4 — MLflow

### 12. What is MLflow?
A tool for tracking machine-learning experiments and storing model artifacts.

### 13. What is logged?
Parameters, MAE, RMSE, R², diagnostic plots and model artifacts.

### 14. Why track experiments?
It makes model experiments reproducible and easier to compare.

### 15. What is reproducibility?
Running the same experiment with the same data and fixed random seed should produce the same result.

### 16. How is reproducibility checked?
The Random Forest is trained twice with `random_state=42` and the RMSE values are compared.

---

# Lab 5 — Production Pipeline

### 17. What is data validation?
Checking that the input dataset has the expected columns, valid dates, non-negative numeric values and required country data.

### 18. What is a scikit-learn Pipeline?
A reusable sequence of transformations that can be fitted on training data and applied consistently to new data.

### 19. What transformations are used?
Median imputation followed by StandardScaler.

### 20. Why fit transformations only on training data?
To prevent information from the test set leaking into training.

### 21. What does output validation check?
It checks for NaNs, matching feature dimensions and matching X/y row counts.

---

## One-minute explanation

"I developed a COVID-19 India regression project that predicts current confirmed cases from the previous day's confirmed cases. The original project cleans the data, performs EDA, uses a chronological train-test split, trains Linear Regression, Decision Tree and Random Forest models, and compares MAE, RMSE and R². I then extended it with three MLOps experiments. Lab 3 converts the workflow into a repeatable baseline pipeline. Lab 4 adds MLflow experiment tracking and a reproducibility check. Lab 5 adds input validation, a production-style scikit-learn preprocessing pipeline, and output validation. This makes the original beginner ML project more reproducible and production-oriented."
