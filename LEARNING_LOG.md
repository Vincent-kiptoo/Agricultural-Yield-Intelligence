# Yield Intelligence: Gradient Boosting for Agricultural Yield Prediction
# Learning Log
## Project Overview

The Yield Intelligence project began as an opportunity to apply machine-learning concepts to a real dataset while developing stronger Python programming, statistical reasoning, software engineering, and machine-learning engineering skills.

The target variable is **Standard Yield**, with environmental, geographical, soil, weather, and crop-related features used as predictors.

The project has progressed from exploratory analysis and linear regression into nonlinear modeling, ensemble learning, model diagnostics, interpretation, and now model deployment.

### Model Progression at a Glance

| Stage                          | RMSE   | R²     |
| ------------------------------ | ------ | ------ |
| Polynomial Regression          | 0.059  | 0.739  |
| Decision Tree                  | 0.0323 | 0.9218 |
| Stacking (DT + RF + GB, Ridge) | 0.0218 | 0.9643 |
| **Gradient Boosting (final)**  | **0.0148** | **0.9836** |

The steady drop in RMSE across stages tells the real story of this project: each modeling choice was a direct response to a limitation exposed by the one before it.

---

## 1. Data Understanding and Exploration

I began by understanding the structure of the processed Maji Ndogo dataset and investigating the relationships between the predictors and Standard Yield.

### Concepts learned

* Feature selection
* Numerical vs categorical variables
* Distribution analysis
* Correlation analysis
* Hypothesis formulation
* Normality testing
* Outlier investigation
* Nonlinear relationships
* Exploratory data analysis

### Key realization

The relationship between several predictors and Standard Yield was not purely linear.

Features such as rainfall, soil fertility, pH, temperature, elevation, and slope suggested nonlinear behavior.

This became an important turning point in the project because it showed that relying only on traditional linear regression would probably limit predictive performance.

---

## 2. Linear Regression

I started with linear regression models to establish a baseline.

I learned:

* Train/test splitting
* Linear regression
* R²
* RMSE
* Residuals
* Model coefficients
* Multicollinearity
* Model assumptions

The initial linear models provided a useful baseline but did not capture all the nonlinear relationships present in the data.

### Important lesson

A baseline model is not necessarily supposed to be the final model.

Its purpose is to give me something against which more sophisticated models can be compared.

---

## 3. Regularization

I then explored:

* Ridge Regression
* Lasso Regression
* Cross-validation
* Hyperparameter selection
* Coefficient shrinkage
* Feature selection through Lasso

I learned that regularization can improve generalization and control model complexity.

Lasso was particularly useful for understanding how irrelevant or weak predictors can have their coefficients reduced to zero.

---

## 4. Polynomial Regression

Because the EDA suggested nonlinear relationships, I experimented with polynomial regression.

The polynomial transformation increased the feature space significantly and allowed linear regression to model nonlinear relationships.

The model achieved approximately:

* **RMSE: 0.059**
* **R²: 0.739**

### Key lesson

Polynomial regression demonstrated that nonlinear relationships existed, but increasing feature complexity also increases the risk of overfitting and computational cost.

This motivated the transition toward tree-based models.

---

## 5. Decision Trees

I then moved into Decision Tree Regression.

The baseline Decision Tree achieved approximately:

* **RMSE: 0.0323**
* **R²: 0.9218**

This was a major improvement over the linear and polynomial approaches.

I learned:

* Decision tree structure
* Recursive splitting
* Tree depth
* Leaf nodes
* `min_samples_leaf`
* `min_samples_split`
* Overfitting
* Underfitting
* Tree complexity

I also experimented with parameter search and cross-validation.

### Important lesson

Decision trees can naturally capture nonlinear relationships and interactions without explicitly creating polynomial features.

---

## 6. Random Forest

I then moved from a single decision tree to Random Forest Regression.

This introduced ensemble learning through:

* Bootstrap sampling
* Multiple decision trees
* Feature randomness
* Aggregation
* Variance reduction

I learned how parameters such as:

* `n_estimators`
* `max_depth`
* `min_samples_leaf`
* `max_features`

affect model performance.

I also used `RandomizedSearchCV` to efficiently search through different combinations of hyperparameters.

### Key lesson

Instead of relying on one high-variance tree, Random Forest combines many trees to produce a more stable prediction.

---

## 7. Gradient Boosting

Gradient Boosting became the strongest individual model in the project.

I experimented with:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `subsample`

Using randomized hyperparameter tuning, I found a strong configuration:

```text
n_estimators       = 661
learning_rate      = 0.168307
max_depth          = 4
min_samples_split  = 18
min_samples_leaf   = 14
subsample          = 0.865009
```

The final Gradient Boosting model achieved approximately:

```text
RMSE: 0.0148
R²:   0.9836
```

This became the selected model for the project.

### Key lesson

Gradient Boosting was able to capture the complex nonlinear relationships in the data while generalizing substantially better than the earlier models.

---

## 8. Stacking Regression

I then experimented with stacking.

The stacking ensemble combined:

* Decision Tree
* Random Forest
* Gradient Boosting

with Ridge as the final estimator.

The stacking model achieved approximately:

```text
RMSE: 0.0218
R²:   0.9643
```

Although stacking performed strongly, it did not outperform the tuned Gradient Boosting model.

### Important lesson

More complex does not automatically mean better.

An ensemble containing several strong models can still perform worse than the best individual model.

Therefore, I selected Gradient Boosting based on empirical performance rather than assuming that stacking would be superior.

---

## 9. Residual Diagnostics

After selecting Gradient Boosting, I moved beyond simply looking at RMSE and R².

I investigated whether the model's errors behaved reasonably.

I examined:

* Residual distribution
* Mean residual
* Residuals vs predictions
* Q-Q behavior
* Heteroscedasticity
* Breusch-Pagan testing
* Error behavior across crop types
* Large individual errors

Key findings:

* Mean residual was approximately **-0.00048**, indicating very little overall bias.
* Residuals were largely centered around zero.
* The Q-Q plot showed some deviation in the tails.
* Breusch-Pagan testing produced a p-value of approximately **0.0697**, so heteroscedasticity was not statistically significant at the 5% level.
* Some crop types showed higher prediction errors than others.
* Tea and cassava had relatively high mean absolute errors.
* A few extreme observations produced substantially larger errors.

### Key lesson

A high R² does not automatically mean a model is reliable.

Residual diagnostics helped me understand **where and how the model makes mistakes**, rather than only measuring how accurate it is overall.

---

## 10. Feature Importance and Model Interpretation

After validating the residual behavior, I investigated what the Gradient Boosting model considered important.

The strongest features included:

1. Rainfall
2. Crop type — Tea
3. pH
4. Latitude
5. Pollution level
6. Elevation
7. Crop type — Rice
8. Crop type — Coffee
9. Soil type — Loamy

### Key insight

Rainfall was the most influential predictor, followed closely by crop-specific and environmental variables.

These results broadly supported patterns identified during EDA.

However, I also learned an important distinction:

> Feature importance indicates how much the model relies on a feature; it does not establish causation or tell me the direction of the relationship.

---

## 11. Python and Software Engineering Development

This project has also significantly changed how I write Python.

Instead of keeping everything inside notebooks, I began extracting reusable functionality into modules.

I learned to:

* Create Python modules
* Use classes
* Build reusable pipelines
* Use imports between modules
* Separate preprocessing from modeling
* Use configurable model parameters
* Create reusable pipeline constructors
* Use logging
* Structure project directories
* Think about separation of concerns

I modularized related models into reusable pipeline components, including:

```text
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor
StackingRegressor
```

This helped me understand that software engineering is not simply about writing more code; it is about **organizing code so that it can be reused, maintained, tested, and extended**.

---

## 12. Git and Version Control

Throughout the project I also became more comfortable with Git.

I learned:

* Branching
* Commits
* Conventional commit messages
* Pull requests
* Merging
* Conflict resolution
* Remote branches
* Cleaning obsolete branches
* Keeping `main` as the central source of truth

I encountered a real merge conflict involving the Gradient Boosting notebook and resolved it instead of discarding the repository state.

### Key lesson

Version control became part of the engineering workflow rather than something used only to back up code.

---

## 13. Current Stage — Model Persistence

The modeling and interpretation stages are now complete.

The next objective is to turn the final Gradient Boosting model into a reusable machine-learning artifact.

The planned workflow is:

```text
Final Gradient Boosting Pipeline
            ↓
           Fit
            ↓
      Save with joblib
            ↓
       Load model
            ↓
    Reproduce predictions
            ↓
       Validate artifact
```

The important principle is that I should save the **entire pipeline**, including preprocessing, rather than only the Gradient Boosting estimator.

This will make the model easier to reuse with raw input data.

---

## 14. Next Stage — Deployment

After validating model persistence, the project will transition from data science into machine-learning engineering.

The planned deployment learning path is:

```text
Model Persistence
       ↓
REST / HTTP Fundamentals
       ↓
FastAPI
       ↓
Pydantic Validation
       ↓
API Testing
       ↓
Docker
       ↓
Cloud Deployment
       ↓
Monitoring
```

The objective is eventually to expose the trained model through a prediction API.

The final architecture will move toward:

```text
User
  ↓
FastAPI
  ↓
Saved Gradient Boosting Pipeline
  ↓
Standard Yield Prediction
```

---

## Overall Learning Reflection

This project has taken me from treating machine learning primarily as a collection of algorithms to understanding it as an **end-to-end engineering process**.

I have learned that building a strong machine-learning solution involves much more than fitting a model.

The complete workflow is:

```text
Understand the problem
        ↓
Explore the data
        ↓
Form hypotheses
        ↓
Build a baseline
        ↓
Experiment with models
        ↓
Tune hyperparameters
        ↓
Compare models
        ↓
Select the final model
        ↓
Diagnose errors
        ↓
Interpret the model
        ↓
Modularize the code
        ↓
Save the model
        ↓
Deploy the model
```

The biggest lesson from Maji Ndogo so far is that **model performance, statistical reasoning, Python programming, software engineering, and deployment are interconnected skills**.

The project has therefore become more than a machine-learning exercise. It is becoming my first complete journey toward building and deploying a production-oriented machine-learning system.