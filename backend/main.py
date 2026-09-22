import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Add parent directory to path so we can import from ml.predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predict import DiseasePredictor

app = FastAPI(title="Disease Prediction API", description="Academic project API for disease prediction based on symptoms")

# Enable CORS for React frontend (Vite default port is usually 5173 or 3000)
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for Render to verify service is alive."""
    return {"status": "ok"}

# Initialize predictor at startup
print("Initializing ML predictor...")
predictor = DiseasePredictor()
print("Predictor initialized successfully.")

class PredictRequest(BaseModel):
    symptoms: List[str]

class PredictionResult(BaseModel):
    disease: str
    confidence: float
    contributing_symptoms: List[str]

@app.get("/symptoms", response_model=List[str])
async def get_symptoms():
    """Returns the full list of valid symptom strings for autocomplete."""
    return predictor.get_all_symptoms()

@app.post("/predict", response_model=List[PredictionResult])
async def predict_disease(request: PredictRequest):
    """
    Accepts a list of symptom strings and returns top 3 predicted diseases.
    Includes confidence percentages and top contributing symptoms per prediction.
    """
    if not request.symptoms:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Please provide at least one symptom.")
        
    # The Predictor handles mapping strings to features and calling the model
    results = predictor.predict(request.symptoms)
    return results
