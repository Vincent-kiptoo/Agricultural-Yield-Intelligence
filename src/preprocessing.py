"""
Contains reusable functions for preparing data and transforming data for machine learming workflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

from src.config import config_params
from src.field_data_processor import FieldDataProcessor
from src.logging_config import get_logger

class ModelPreprocessors:
    def __init__(self):
        self.field_df =FieldDataProcessor(config_params).process()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.num_features = None
        self.cat_features = None
        self.logger = get_logger(__name__)

        self.logger.info("Model preprocessor is initialized")

    def load_data(self) -> pd.DataFrame:
        if self.field_df is None:
            raise ValueError("Field_df is None. Ensure to load the the data to the dataframe")
            self.logger.info("The data is successfuly loaded into pandas DAtaFrame")
        return self.field_df
    
    def X_y_features(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.field_df is None:
            raise ValueError("field df is an empty dataframe")
        features = [col for col in self.field_df.columns if col not in ["Standard_yield", "Annual_yield"]]
        self.X = self.field_df[features]
        self.y = self.field_df["Standard_yield"]
        return self.X, self.y
    
    def X_y_train_test_split(self) -> tuple:
        if self.X is None or self.y is None:
            raise RuntimeError("Call X_y_features() before X_y_train_test_split().")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.logger.info("Train test split is successfully")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_types(self)-> tuple[list[str], list[str]]:
        if self.X_train is None:
            self.X_train, _, _, _, = self.X_y_train_test_split()
        self.num_features = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        self.cat_features = self.X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        return self.num_features, self.cat_features

    def prepare_data(self):
        """Prepare data for model pipelines."""

        self.load_data()
        self.X_y_features()
        self.X_y_train_test_split()
        self.get_feature_types()

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.num_features,
            self.cat_features
        )