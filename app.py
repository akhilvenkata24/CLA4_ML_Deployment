"""
CLA 4 - Task 5: FastAPI Model Serving (app.py)
Description:
    Production-ready REST API for serving the Titanic survival prediction model.
    Loads the saved model pipeline (model.pkl) without retraining.
    Provides GET /health, POST /predict, and automatic interactive Swagger documentation at /docs.
"""

import os
from typing import Optional, Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="FastAPI service for serving the machine learning model pipeline trained on the Titanic Kaggle dataset.",
    version="1.0.0"
)

# Global model container
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model_pipeline = None


@app.on_event("startup")
def load_model():
    """Load the pre-trained model pipeline once during application startup."""
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"[FastAPI] Model loaded successfully from {MODEL_PATH}")


# Pydantic schema for request validation
class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)", example=1)
    Sex: Literal["male", "female"] = Field(..., description="Passenger gender ('male' or 'female')", example="female")
    Age: Optional[float] = Field(None, ge=0.0, le=120.0, description="Age in years", example=28.0)
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard", example=0)
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard", example=0)
    Fare: float = Field(..., ge=0.0, description="Passenger fare paid", example=80.0)
    Embarked: Optional[Literal["S", "C", "Q"]] = Field(None, description="Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)", example="C")

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 1,
                "Sex": "female",
                "Age": 28.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 82.50,
                "Embarked": "C"
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Did Not Survive, 1 for Survived")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    survival_probability: float = Field(..., description="Probability of survival (0 to 100%)")
    input_data: dict = Field(..., description="Echo of input passenger features")


@app.get("/", tags=["General"])
def root():
    """Root endpoint welcoming users and pointing to API documentation."""
    return {
        "message": "Welcome to the Titanic Survival Prediction API",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint reporting server and model status."""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_file": os.path.basename(MODEL_PATH),
        "model_type": type(model_pipeline.named_steps["classifier"]).__name__ if model_pipeline else None,
        "features": ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(passenger: PassengerInput):
    """
    Online prediction endpoint.
    Accepts passenger features, processes them through the fitted pipeline,
    and returns survival prediction and probability.
    """
    global model_pipeline
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    try:
        # Convert Pydantic model to DataFrame matching training columns
        input_dict = passenger.dict()
        df_input = pd.DataFrame([input_dict])

        # Predict using the loaded pipeline
        pred = int(model_pipeline.predict(df_input)[0])
        prob = float(model_pipeline.predict_proba(df_input)[0][1] * 100)

        label = "Survived" if pred == 1 else "Did Not Survive"

        return PredictionResponse(
            prediction=pred,
            prediction_label=label,
            survival_probability=round(prob, 2),
            input_data=input_dict
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
