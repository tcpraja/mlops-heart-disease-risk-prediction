from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from api.schema import HeartDiseaseInput, PredictionResponse
from src.config import MODEL_PATH
from src.inference import predict_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("heart_disease_api")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
PREDICTION_COUNT = Counter("model_predictions_total", "Total model predictions", ["prediction"])
ERROR_COUNT = Counter("api_errors_total", "Total API errors", ["endpoint"])
LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="FastAPI inference service for the UCI Heart Disease MLOps assignment.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.time() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, str(status_code)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        LOGGER.info(
            "method=%s endpoint=%s status=%s latency_seconds=%.4f",
            request.method,
            endpoint,
            status_code,
            elapsed,
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Heart Disease Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_artifact_missing",
        "model_artifact_found": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseInput) -> PredictionResponse:
    if not MODEL_PATH.exists():
        ERROR_COUNT.labels("/predict").inc()
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `python src/train.py` before serving predictions.",
        )
    try:
        result = predict_one(payload.model_dump())
        PREDICTION_COUNT.labels(str(result["prediction"])).inc()
        LOGGER.info(
            "prediction=%s probability_heart_disease=%s",
            result["prediction"],
            result["probability_heart_disease"],
        )
        return PredictionResponse(**result)
    except Exception as exc:
        ERROR_COUNT.labels("/predict").inc()
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
