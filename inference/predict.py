import joblib
from pathlib import Path
from typing import cast


import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gradient_boosting_pipeline.joblib"
)


def load_model()-> Pipeline:
    model = joblib.load(MODEL_PATH)
    return cast(Pipeline, model)


def predict(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    prediction = model.predict(data)
    return cast(np.ndarray, prediction)