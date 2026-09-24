# Agricultural Yield Intelligence

**An end-to-end machine learning system for agricultural yield prediction** covering data engineering, statistical modelling, ensemble learning, model evaluation, interpretability, and deployment.

Agricultural productivity depends on a web of interacting environmental, geographical, soil, weather, and crop factors. This project investigates those relationships and builds a model that estimates **Standard Yield**, with the goal of giving farmers and agricultural decision-makers an early, data-driven number to plan around.

This project uses synthetic agricultural data and is intended to demonstrate an end-to-end machine learning workflow. The reported performance should not be interpreted as evidence of performance on real world agricultural data.

![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Model](https://img.shields.io/badge/model-Gradient%20Boosting-blue)
![R²](https://img.shields.io/badge/R²-0.984-brightgreen)
![Framework](https://img.shields.io/badge/API-FastAPI-009688)

## Table of Contents

- [Objectives](#objectives)
- [Workflow](#workflow)
- [Model Comparison](#model-comparison)
- [Final Model](#final-model)
- [Model Diagnostics](#model-diagnostics)
- [Debugging Highlights](#debugging-highlights)
- [Feature Importance](#feature-importance)
- [Model Persistence & Deployment](#model-persistence--deployment)
- [Repository Structure](#repository-structure)
- [Engineering Practices](#engineering-practices)
- [Project Status](#project-status)
- [Key Findings](#key-findings)
- [What This Project Taught Me](#what-this-project-taught-me)

---

## Objectives

- Understand which factors drive Standard Yield, and how strongly
- Investigate linear vs. nonlinear relationships in the data
- Establish baseline regression models, then beat them with tree-based ensembles
- Tune hyperparameters and diagnose residuals rather than trusting R² alone
- Interpret feature importance without overclaiming causality
- Package preprocessing + model as one reusable, serializable pipeline
- Serve the model as a prediction API, moving toward full deployment

---

## Workflow

```
Data Ingestion → Data Processing → EDA → Feature Analysis → Train/Test Split → Preprocessing Pipeline → Baseline Models → Polynomial Regression
      → Decision Tree → Random Forest → Gradient Boosting → Stacking → Hyperparameter Tuning → Model Comparison → Residual Diagnostics
      → Feature Importance → Final Model → Persistence → Inference → API Serving → Testing → Containerization → Cloud Deployment
```

---

## Model Comparison

Eight regression approaches were evaluated, from linear baselines through boosted ensembles — with a clear nonlinear signal in the data favoring tree-based methods.

| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression (OLS) | 0.0678 | 0.6553 |
| Ridge (RidgeCV) | 0.0678 | 0.6552 |
| Lasso (LassoCV) | 0.0699 | 0.6333 |
| Polynomial Regression | 0.0589 | 0.7396 |
| Decision Tree | 0.0322 | 0.9223 |
| Random Forest | 0.0208 | 0.9674 |
| **Gradient Boosting (final)** | **0.0148** | **0.9836** |
| Stacking Regressor | 0.0218 | 0.9643 |

---

## Final Model

**Gradient Boosting Regressor**, tuned and wrapped in a scikit-learn `Pipeline` alongside preprocessing:

```
Raw Features → ColumnTransformer (numeric passthrough + one-hot categoricals) → Gradient Boosting Regressor → Standard Yield Prediction
```

**Tuned hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 661 |
| `learning_rate` | 0.168307 |
| `max_depth` | 4 |
| `min_samples_split` | 18 |
| `min_samples_leaf` | 14 |
| `subsample` | 0.865009 |
| `random_state` | 42 |

**Performance:** RMSE = 0.0148, R² = 0.9836 — the model explains ~98.4% of the variance in Standard Yield on held-out data.

> Standard Yield is scaled to 0–1, so RMSE reads relative to that scale, not as a raw-unit error.

---

## Model Diagnostics

High R² doesn't earn trust on its own — residuals were checked for bias, distribution, variance, normality, heteroscedasticity, extreme errors, and consistency across crop types.

- **Mean residual:** ≈ −0.00048 — negligible overall bias
- **Breusch-Pagan test:** p = 0.0697 — not significant at 5%, but close enough to keep monitoring
- Larger errors observed in the distribution tails and unevenly across certain crop types

---

## Debugging Highlights

Two correctness bugs were caught during code audit, before they could distort results:

- **Whitespace-ordering bug in `apply_correction`** — the correction mapping ran before crop-name strings were stripped, so padded names silently skipped correction
- **Sort-before-`abs()` bug in `target_correlation`** — correlations were sorted on signed values before taking the absolute value, misranking feature importance

Both were caught through direct testing of the pipeline modules and fixed before finalizing the model.

---

## Feature Importance

Ranked by model-dependence (not causal influence):

1. Rainfall
2. Crop type — Tea
3. pH
4. Latitude
5. Pollution level
6. Elevation
7. Crop type — Rice
8. Crop type — Coffee
9. Soil type — Loamy

**Rainfall is the dominant predictor.** A feature ranking high here means the model relies on it — not that changing it would causally move yield.

---

## Model Persistence & Deployment

The full trained pipeline (preprocessing + estimator) is serialized with `joblib`, so the deployment environment never has to manually recreate feature engineering:

```
Raw Input → ColumnTransformer → Gradient Boosting Regressor → Prediction
```

A reusable inference module (`inference/predict.py`) exposes `load_model()` and `predict()`, and the model is loaded once at startup rather than reloaded per request.

The prediction service is served through **FastAPI**:

```
GET  /          → health check
POST /predict   → Standard Yield prediction
```

**Request** (15 farm-level features, validated via Pydantic):

```json
{
  "Elevation": 786.0558,
  "Latitude": -7.389911,
  "Longitude": -7.556202,
  "Location": "Rural_Akatsi",
  "Slope": 14.795113,
  "Rainfall": 1125.2,
  "Min_temperature_C": -3.1,
  "Max_temperature_C": 33.1,
  "Ave_temps": 15.0,
  "Soil_fertility": 0.62,
  "Soil_type": "Sandy",
  "pH": 6.169393,
  "Pollution_level": 0.085267,
  "Plot_size": 1.3,
  "Crop_type": "cassava"
}
```

**Response:**

```json
{ "prediction": 0.57372716 }
```

**Architectur:e** HTTP serving is kept separate from inference logic so the model can be reused outside the API:

```
Client → api.py (FastAPI, Pydantic, DataFrame)
       → predict.py (load_model(), predict())
       → gradient_boosting_pipeline.joblib → Prediction
```

Currently tested locally via FastAPI's Swagger UI (`/docs`). Containerization and cloud deployment are the next milestones — see [Project Status](#project-status).

---

## Repository Structure

```
Agricultural-Yield-Intelligence/
├── config/            # configuration files
├── gradient_booster/  # final model artifacts / boosting-specific code
├── inference/         # api.py, predict.py — model serving
├── model_pipelines/   # ensemble_models.py, tree_pipeline.py
├── notebooks/         # 01_eda.ipynb → 10_feature_importance.ipynb
├── reports/           # diagnostics, evaluation outputs
├── src/               # config.py, data_ingestion.py, preprocessing.py,
│                       # logging_config.py, field_data_processor.py
├── .gitignore
├── LEARNING_LOG.md
├── requirements.txt
└── README.md
```

---

## Engineering Practices

- Reusable, modular Python (type hints, logging, config-driven parameters)
- scikit-learn `Pipeline` + `ColumnTransformer` for coupled preprocessing/modeling
- Git version control with a running `LEARNING_LOG.md`
- Model persistence via `joblib`
- FastAPI + Pydantic for validated model serving
- Hyperparameter tuning, residual diagnostics, and feature-importance interpretation as standard steps, not afterthoughts

---

## Project Status

**Current stage:** local API serving and deployment engineering

**Done**
- [x] Data processing, EDA, feature analysis
- [x] Full model comparison (linear → ensemble)
- [x] Hyperparameter tuning, residual diagnostics, feature importance
- [x] Pipeline persistence and reusable inference module
- [x] Local FastAPI prediction API with Pydantic validation
- [x] Verified end-to-end local prediction

---

## Key Findings

1. Standard Yield has substantial nonlinear relationships with its predictors.
2. Polynomial regression beat linear baselines but lost to every tree-based model.
3. Ensemble methods delivered the biggest performance jump.
4. Tuned Gradient Boosting outperformed even the stacked ensemble.
5. Residual bias is negligible on average, though some crop types and extreme cases still show larger errors.
6. Rainfall, crop type, pH, latitude, pollution, and elevation are the top drivers.
7. A high R² is a starting point, not a finish line — diagnostics and interpretation still matter.

---

## What This Project Taught Me

- **Data science:** EDA, statistical testing, regression, cross-validation, hyperparameter tuning, residual diagnostics, model interpretation
- **Machine learning:** linear models through gradient boosting and stacking, preprocessing pipelines, model persistence, single vs. batch inference
- **Python & software engineering:** OOP, modular packages, type hints, logging, Git workflows, REST APIs, Pydantic validation, FastAPI model serving

---

*This project is an ongoing transition from data science experimentation into machine learning engineering. The goal isn't just a high-performing model, but understanding every layer of how it becomes a usable service.*
