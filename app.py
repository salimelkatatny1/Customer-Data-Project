from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load regression model, preprocessing pipeline, and MAE
model = joblib.load("best_regressor.pkl")
preprocess = joblib.load("preprocess.pkl")
mae = joblib.load("regression_mae.pkl")

class CustomerData(BaseModel):
    age: int
    gender: str
    income: float
    education: str
    region: str
    loyalty_status: str
    purchase_frequency: str
    product_category: str
    promotion_usage: int
    satisfaction_score: float

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    
    X_prep = preprocess.transform(df)
    pred = float(model.predict(X_prep)[0])

    # =========================
    # Confidence estimation
    # =========================
    # Relative error-based confidence
    confidence = max(0.0, 1 - (mae / (abs(pred) + 1e-6)))
    confidence = min(confidence, 1.0)

    return {
        "predicted_purchase_amount": round(pred, 2),
        "expected_error": round(float(mae), 2),
        "confidence_score": round(confidence, 3)
    }
