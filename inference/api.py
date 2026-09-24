from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from inference.predict import load_model, predict as make_prediction


app = FastAPI()

model = load_model()


class PredictionRequest(BaseModel):
    Elevation: float
    Latitude: float
    Longitude: float
    Location: str
    Slope: float
    Rainfall: float
    Min_temperature_C: float
    Max_temperature_C: float
    Ave_temps: float
    Soil_fertility: float
    Soil_type: str
    pH: float
    Pollution_level: float
    Plot_size: float
    Crop_type: str


@app.get("/")
def home():
    return {"message": "Yield Intelligence API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame([request.model_dump()])
    prediction = make_prediction(model, data)
    print(prediction)
    return {"prediction": float(prediction[0])}