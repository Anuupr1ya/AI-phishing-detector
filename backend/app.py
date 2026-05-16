from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_model.pkl")

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully ✅")
print(f"Model expects {model.n_features_in_} features")

app = FastAPI(title="AI Phishing Detector API")

class URLFeatures(BaseModel):
    features: list

@app.get("/")
def home():
    return {
        "message": "AI Phishing Detector is running ✅",
        "accuracy": "96.98%",
        "features_required": model.n_features_in_
    }

@app.post("/predict")
def predict(data: URLFeatures):
    try:
        features = np.array(data.features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return {
            "prediction": "PHISHING 🚨" if prediction == 1 else "LEGITIMATE ✅",
            "confidence": f"{max(probability) * 100:.2f}%",
            "phishing_probability": f"{probability[1] * 100:.2f}%",
            "legitimate_probability": f"{probability[0] * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})