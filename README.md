# Yield Intelligence: Gradient Boosting for Agricultural Yield Prediction

An end-to-end machine learning project for predicting **Standard Yield** from agricultural, environmental, geographical, soil, weather, and crop-related features.

The project explores the complete machine learning workflow, from data preparation and exploratory analysis to model development, hyperparameter tuning, diagnostics, interpretation, and model deployment.


## Project Overview

Agricultural productivity is influenced by multiple interacting environmental and geographical factors. The objective of this project is to investigate these relationships and develop a machine learning model capable of accurately predicting Standard Yield.

The project began with statistical and exploratory analysis before progressing through increasingly complex regression models, ending with a tuned Gradient Boosting Regressor as the final selected model.


## Objectives

The project aims to:

* Understand the factors associated with Standard Yield.
* Investigate linear and nonlinear relationships in the data.
* Establish baseline regression models.
* Develop and compare nonlinear tree-based models.
* Optimize model hyperparameters.
* Diagnose model residuals and potential sources of error.
* Interpret model feature importance.
* Build reusable preprocessing and modeling pipelines.
* Save the final trained model as a reusable artifact.
* Deploy the model as a prediction service.


## Machine Learning Workflow

```text
Data Ingestion
      ↓
Data Processing
      ↓
Exploratory Data Analysis
      ↓
Feature Analysis
      ↓
Train/Test Split
      ↓
Preprocessing Pipeline
      ↓
Baseline Models
      ↓
Polynomial Regression
      ↓
Decision Tree
      ↓
Random Forest
      ↓
Gradient Boosting
      ↓
Stacking
      ↓
Hyperparameter Tuning
      ↓
Model Comparison
      ↓
Residual Diagnostics
      ↓
Feature Importance
      ↓
Final Model
      ↓
Model Persistence
      ↓
Deployment
```


## Models Evaluated

The project evaluated several regression approaches.

| Model                       | Purpose                              |
| --------------------------- | ------------------------------------ |
| Linear Regression           | Baseline                             |
| Ridge Regression            | Regularized linear baseline          |
| Lasso Regression            | Regularization and feature selection |
| Polynomial Regression       | Capture nonlinear relationships      |
| Decision Tree Regressor     | Nonlinear tree-based model           |
| Random Forest Regressor     | Bagging ensemble                     |
| Gradient Boosting Regressor | Boosting ensemble                    |
| Stacking Regressor          | Combine multiple models              |

The experiments demonstrated that the dataset contains substantial nonlinear structure, making tree-based ensemble methods particularly effective.


## Final Model

The final selected model is a **Gradient Boosting Regressor**.

After hyperparameter tuning, the selected configuration was:

```text
n_estimators       = 661
learning_rate      = 0.168307
max_depth          = 4
min_samples_split  = 18
min_samples_leaf   = 14
subsample          = 0.865009
```

### Performance

```text
RMSE = 0.0148
R²   = 0.9836
```

The model explains approximately **98.4% of the variation in Standard Yield** on the evaluation data. Standard Yield is [normalized to a 0–1 scale / measured in <units> — confirm and fill in], so the RMSE above should be interpreted relative to that scale rather than as an absolute error in raw yield units.


## Model Diagnostics

Model evaluation was not limited to RMSE and R².

Residual diagnostics were performed to investigate:

* Prediction bias
* Residual distribution
* Residual variance
* Normality behavior
* Heteroscedasticity
* Extreme prediction errors
* Error differences across crop types

The mean residual was approximately:

```text
-0.00048
```

indicating very little overall prediction bias.

The Breusch-Pagan test produced:

```text
p-value = 0.0697
```

Therefore, heteroscedasticity was not statistically significant at the 5% significance level, although the result is close enough to the threshold to warrant continued monitoring.

The model also showed some larger errors in the tails of the residual distribution and uneven error rates across certain crop types.


## Debugging Highlights

The code audit phase surfaced two correctness bugs that were caught before they could distort downstream results:

* **Whitespace-ordering bug in `apply_correction`** — the correction mapping was applied before crop-name strings were stripped, so padded crop names silently failed to match and skipped correction.
* **Sort-before-`abs()` bug in `target_correlation`** — correlations were sorted on signed values before taking the absolute value, producing a misleading ranking of feature importance by correlation strength.

Both were identified through direct sandbox testing of the pipeline modules and fixed prior to model finalization.


## Feature Importance

Feature importance analysis identified several influential predictors.

The most important features included:

1. Rainfall
2. Crop type — Tea
3. pH
4. Latitude
5. Pollution level
6. Elevation
7. Crop type — Rice
8. Crop type — Coffee
9. Soil type — Loamy

Rainfall was the most influential feature according to the model's feature-importance estimates.

However, feature importance should be interpreted as **model dependence rather than causal influence**.


## Project Structure

```text
maji-ndogo-yield-intelligence/
│
├── data/
│
├── models/
│
├── model_pipelines/
│   ├── __init__.py
│   ├── ensemble_models.py
│   ├── tree_pipeline.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_ols.ipynb
│   ├── 03_regularization.ipynb
│   ├── 04_polynomial_regression.ipynb
│   ├── 05_decision_tree.ipynb
│   ├── 06_random_forest.ipynb
│   ├── 07_gradient_boosting.ipynb
│   ├── 08_stacking.ipynb
│   ├── 09_residual_diagnostics.ipynb
│   └── 10_feature_importance.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── logging_config.py
│   ├── preprocessing.py
│   ├── field_data_processor.py
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

*The structure will evolve as testing, model persistence, and deployment are added.*


## Engineering Practices

The project emphasizes reproducibility and maintainability through:

* Reusable Python modules
* Scikit-learn pipelines
* Separation of preprocessing and modeling
* Configurable model parameters
* Logging
* Git version control
* Hyperparameter tuning
* Model diagnostics
* Model persistence

Unit testing is planned but not yet implemented — see Project Status below.


## Model Persistence

The final trained pipeline will be serialized using `joblib`.

The entire pipeline will be saved rather than only the estimator so that preprocessing and prediction remain coupled.

```text
Raw Input
   ↓
Preprocessing
   ↓
Gradient Boosting Model
   ↓
Prediction
```

This allows the saved artifact to be loaded later without rebuilding the preprocessing workflow manually.


## Deployment

The next stage of the project is to expose the trained model through an API.

The planned architecture is:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Saved ML Pipeline
  │
  ▼
Standard Yield Prediction
```

Deployment technologies will include:

* FastAPI
* Pydantic
* pytest
* Docker
* Cloud deployment


## Key Findings

The project produced several important findings:

1. Standard Yield contains substantial nonlinear relationships with the available predictors.
2. Polynomial regression improved upon the initial linear models but remained inferior to tree-based approaches.
3. Decision trees captured nonlinear relationships effectively but required complexity control.
4. Ensemble methods substantially improved predictive performance.
5. Gradient Boosting produced the strongest individual predictive performance.
6. Stacking did not outperform the tuned Gradient Boosting model.
7. Residual diagnostics showed very little overall bias.
8. Some crop types and extreme observations produced larger prediction errors.
9. Rainfall, crop type, pH, latitude, pollution, and elevation were among the most influential predictors.
10. High predictive performance does not eliminate the need for residual diagnostics and model interpretation.


## What I Learned

This project developed skills across several areas:

### Data Science

* Exploratory data analysis
* Statistical testing
* Regression
* Model evaluation
* Cross-validation
* Hyperparameter tuning
* Residual diagnostics
* Model interpretation

### Machine Learning

* Linear models
* Regularization
* Polynomial regression
* Decision trees
* Random forests
* Gradient boosting
* Stacking
* Ensemble learning

### Python

* Object-oriented programming
* Modules and packages
* Reusable functions
* Classes
* Scikit-learn pipelines
* Logging
* Testing
* Project structure

### Software Engineering

* Git and GitHub
* Branching and merging
* Conflict resolution
* Modular architecture
* Reproducibility
* Model persistence


## Project Status

**Current stage: Testing, model finalization, and deployment preparation**

Completed:

* [x] Data processing
* [x] Exploratory data analysis
* [x] Baseline regression
* [x] Regularization
* [x] Polynomial regression
* [x] Decision tree
* [x] Random forest
* [x] Gradient boosting
* [x] Stacking
* [x] Hyperparameter tuning
* [x] Model comparison
* [x] Residual diagnostics
* [x] Feature importance
* [x] Pipeline modularization

In progress:

* [ ] Add unit tests for `src/` modules
* [ ] Save final Gradient Boosting pipeline
* [ ] Validate serialized model
* [ ] Build prediction API
* [ ] Add API tests
* [ ] Containerize application
* [ ] Deploy model
* [ ] Add basic monitoring


This project represents an ongoing transition from data science toward machine learning engineering, combining statistical analysis, machine learning, Python programming, software engineering, and deployment.