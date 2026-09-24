"""
This module trains the final gradient boosting regressor model
that predicts the standard yield with RMSE of 0.0148 and R-squared of 0.9836.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import cast
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from src.preprocessing import ModelPreprocessors
from src.logging_config import get_logger


class GradientBooster:
    def __init__(self) -> None:
        self.mp = ModelPreprocessors()
        self.logger = get_logger(__name__)
        self.field_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.pipeline_transformer = None
        self.gradient_pipeline = None
        self.boosting_param = None
        self.fitted_search = None
        self.best_model: Pipeline | None = None

        self.logger.info("GradientBooster initialized")


    def load_data(self) -> pd.DataFrame:
        "Loads the raw data for downstream use"
        self.field_df = self.mp.load_data()
        return self.field_df

    def features_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        "Splits data into training and testing sets"
        if self.mp.X is None or self.mp.y is None:
            self.mp.X_y_features()
        self.X_train, self.X_test, self.y_train, self.y_test = self.mp.X_y_train_test_split()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def boosting_transformer(self) -> ColumnTransformer:
        "Transforms only the categorical features into binary via OneHotEncoder"
        if self.mp.X is None or self.mp.y is None:
            self.mp.X_y_features()
        num_features, cat_features = self.mp.get_feature_types()
        self.pipeline_transformer = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )
        return self.pipeline_transformer

    def boosting_pipeline(self) -> Pipeline:
        "Builds the full preprocessing + model pipeline"
        if self.pipeline_transformer is None:
            raise RuntimeError("Call boosting_transformer() before boosting_pipeline().")
        self.gradient_pipeline = Pipeline(
            steps=[
                ("transformer", self.pipeline_transformer),
                ("model", GradientBoostingRegressor(random_state=42))
            ]
        )
        return self.gradient_pipeline

    def gradient_boosting_params(self) -> dict:
        "Defines the hyperparameter search space"
        self.boosting_param = {
            "model__n_estimators": randint(100, 800),
            "model__learning_rate": uniform(0.01, 0.19),
            "model__max_depth": randint(2, 8),
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 15),
            "model__subsample": uniform(0.6, 0.4)
        }
        return self.boosting_param

    def hyperparameter_search(self) -> RandomizedSearchCV:
        "Builds but does not yet fit the randomized search"
        if self.gradient_pipeline is None:
            raise RuntimeError("Call boosting_pipeline() before hyperparameter_search().")
        if self.boosting_param is None:
            raise RuntimeError("Call gradient_boosting_params() before hyperparameter_search().")
        return RandomizedSearchCV(
            estimator=self.gradient_pipeline,
            param_distributions=self.boosting_param,
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=2
        )


    def train(self) -> Pipeline:
        """
        Runs the full pipeline end-to-end: load data, split, build the
        transformer + model pipeline, search hyperparameters, fit, and
        store the best fitted pipeline on self.best_model.

        Returns the fitted best-performing Pipeline.
        """
        self.logger.info("Starting model training run")

        if self.X_train is None:
            raise RuntimeError("Run feature_split() first")
        self.load_data()
        self.features_split()
        self.boosting_transformer()
        self.boosting_pipeline()
        self.gradient_boosting_params()
        random_search = self.hyperparameter_search()

        self.logger.info("Fitting RandomizedSearchCV (this may take a while to complete)")

        self.fitted_search = random_search.fit(self.X_train, self.y_train)

        self.best_model = cast(Pipeline, self.fitted_search.best_estimator_)

        self.logger.info(
            "Training complete. Best CV RMSE: %.4f",
            -self.fitted_search.best_score_,
        )
        return self.best_model


    def evaluate(self) -> dict:
        """
        Evaluates self.best_model on the held-out test set.
        Must be called after train().
        """
        if self.best_model is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Call train() before evaluate().")

        y_pred = self.best_model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        self.logger.info("Test RMSE: %.4f | Test R2: %.4f", rmse, r2)
        return {"rmse": rmse, "r2 score": r2}


    def save_model(self, path: str = "models/gradient_boosting_pipeline.joblib") -> str:
        """
        Saves the fitted best_model pipeline to disk with joblib.
        Must be called after train().
        """
        if self.best_model is None:
            raise RuntimeError("Call train() before save().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, output_path)
        self.logger.info("Model saved to %s", output_path)
        return str(output_path)


if __name__ == "__main__":
    booster = GradientBooster()
    booster.train()
    metrics = booster.evaluate()
    booster.save_model()

    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test R2:   {metrics['r2']:.4f}")
