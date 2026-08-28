"""
Contains reusable pipelines for RandomForestRegressor, 
GradientBoostingRegressor and StackingRegressor
"""

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    StackingRegressor)

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.logging_config import get_logger
from src.preprocessing import ModelPreprocessors
from model_pipelines.tree_pipelines import TreePipeline

class EnsembleStacking:
    def __init__(self)->None:
        self.module_params = ModelPreprocessors()
        self.transformer = None
        self.logger = get_logger(__name__)

        self.logger.info("RandomForestRegressor is initialized")

    def prepare_data(self):
        """Prepare data and identify numerical and categorical features."""
        self.module_params.prepare_data()

    def feature_transfromer(self) -> ColumnTransformer:
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.module_params.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.module_params.cat_features)
            ]
        )
        return self.transformer

    def random_forest_pipeline(self, **forest_params) -> Pipeline:
        self.forest_pipeline = Pipeline(
            steps=[
                ("transformer", self.transformer),
                ("model", RandomForestRegressor(random_state=42, **forest_params))
            ]
        )
        return self.forest_pipeline

    def gradient_boosting_pipeline(self, **gradient_params) -> Pipeline:
        self.gradient_pipeline = Pipeline(
            steps=[
                ("transformer", self.transformer),
                ("model", GradientBoostingRegressor(random_state=42, **gradient_params))
            ]
        )
        return self.gradient_pipeline

    def stacking_regressor_pipeline(self, **stacking_params)-> StackingRegressor:
        decision_tree = TreePipeline()
        decision_tree.prepare_data()
        decision_tree.tree_feature_transformation()
        tree_pipeline = decision_tree.decision_tree_pipeline()

        estimators = [
            ("tree", tree_pipeline),
            ("forest", self.random_forest_pipeline()),
            ("gradient", self.gradient_boosting_pipeline())
        ]
        stacking_model = StackingRegressor(

                    estimators=estimators,
                    final_estimator=Ridge(),
                    cv=5,
                    n_jobs=2,
                    **stacking_params                )

        return stacking_model