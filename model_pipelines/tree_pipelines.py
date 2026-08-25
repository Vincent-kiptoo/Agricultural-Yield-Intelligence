"""Contains reuseable pipeline for DecisionTreeRegressor"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import ModelPreprocessors
from src.logging_config   import get_logger

class TreePipeline:
    def __init__(self)-> None:
        self.module_params = ModelPreprocessors()
        self.transformer = None
        self.pipeline = None
        self.logger = get_logger(__name__)

        self.logger.info("Tree pipeline is initialized")

    def prepare_data(self):
        """Prepare data and identify numerical and categorical features."""
        self.module_params.prepare_data()

    def tree_feature_transformation(self)-> ColumnTransformer:
        """Create a Decision Tree regression pipeline."""
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.module_params.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.module_params.cat_features)
            ]
        )
        return self.transformer



    def decision_tree_pipeline(self, **tree_params)-> Pipeline:
        self.pipeline = Pipeline(
            steps=[
                ("transformer", self.transformer),
                ("model", DecisionTreeRegressor(random_state=42, **tree_params))
            ]
        )
        return self.pipeline