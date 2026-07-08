from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HeartDiseaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type")
    trestbps: float = Field(..., ge=60, le=260, description="Resting blood pressure")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: float = Field(..., ge=0, le=7, description="Thalassemia status")


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    confidence: float
    probability_heart_disease: float
    model_name: str
    model_version: str
