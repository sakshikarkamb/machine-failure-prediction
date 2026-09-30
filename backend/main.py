import os
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import sys

# Locate directories relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
for directory in [BASE_DIR, BACKEND_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from backend.schemas import (
        PredictionInput,
        PredictionResponse,
        HealthResponse,
        RootResponse
    )
except ImportError:
    from schemas import (
        PredictionInput,
        PredictionResponse,
        HealthResponse,
        RootResponse
    )

# Initialize FastAPI application
app = FastAPI(
    title="Machine Failure Prediction API",
    description="Production-ready FastAPI backend for predictive maintenance using a tuned Random Forest pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate saved model relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "machine_failure_random_forest.pkl"

# Global model container
model = None

def load_saved_model():
    """Load the trained machine failure Random Forest pipeline from disk."""
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Successfully loaded model pipeline from: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model pipeline: {e}")
        raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")

# Load model immediately upon startup
try:
    load_saved_model()
except Exception as e:
    print(f"[CRITICAL] Model initialization failed: {e}")


@app.get("/", response_model=RootResponse, tags=["General"])
def read_root():
    """Root endpoint verifying API availability."""
    return RootResponse(message="Machine Failure Prediction API is running")


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint verifying API and model status."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Predict machine failure based on sensor and machine parameters.
    
    Expected features:
    - Type ('L', 'M', 'H')
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available."
        )

    try:
        # Construct DataFrame matching exact feature names expected by the pipeline
        data_dict = {
            "Type": [input_data.type.upper()],
            "Air temperature [K]": [float(input_data.air_temperature)],
            "Process temperature [K]": [float(input_data.process_temperature)],
            "Rotational speed [rpm]": [float(input_data.rotational_speed)],
            "Torque [Nm]": [float(input_data.torque)],
            "Tool wear [min]": [float(input_data.tool_wear)],
        }
        input_df = pd.DataFrame(data_dict)

        # Generate model prediction and class probabilities
        pred_label = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        failure_prob = float(probabilities[1])

        result_text = "Machine Failure" if pred_label == 1 else "No Machine Failure"

        return PredictionResponse(
            prediction=pred_label,
            result=result_text,
            failure_probability=round(failure_prob, 4)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(exc)}"
        )


@app.get("/model-info", tags=["Analytics"])
def get_model_info() -> Dict[str, Any]:
    """Retrieve model metadata and extracted feature importances."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        
        feature_names = list(preprocessor.get_feature_names_out())
        raw_importances = list(classifier.feature_importances_)
        
        # Clean feature names for clean UI display
        cleaned_importances = {}
        for name, val in zip(feature_names, raw_importances):
            clean_name = name.replace("categorical__", "").replace("numerical__", "")
            cleaned_importances[clean_name] = round(float(val), 4)

        # Sort descending by importance
        sorted_importances = dict(
            sorted(cleaned_importances.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "model_type": type(classifier).__name__,
            "pipeline_steps": [name for name, _ in model.steps],
            "feature_importances": sorted_importances,
            "classes": [int(c) for c in classifier.classes_]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract model info: {str(exc)}"
        )
