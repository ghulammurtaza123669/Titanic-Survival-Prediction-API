# ============================================================
# FastAPI App for Titanic Survival Prediction
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# ---------------------- Load the trained model ----------------------
model = joblib.load("titanic_best_model.joblib")

# ---------------------- Create FastAPI instance ----------------------
app = FastAPI(title="Titanic Survival Prediction API",
              description="Predict survival chances using trained ML model",
              version="1.0")

# ---------------------- Define Input Schema ----------------------
class PassengerData(BaseModel):
    Pclass: int
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    FamilySize: int
    IsAlone: int
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

# ---------------------- Define Prediction Route ----------------------
@app.post("/predict")
def predict(data: PassengerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict probability & class
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    return {
        "Survived": int(pred),
        "Survival_Probability": round(float(prob), 3)
    }

# ---------------------- Root Endpoint ----------------------
@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running 🚀"}