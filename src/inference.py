from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH, MODEL_VERSION


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}. Run `python src/train.py`.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
    return {"model_name": "unknown", "model_version": MODEL_VERSION}


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    metadata = load_metadata()
    input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= 0.5)
    label = "Heart disease risk present" if prediction == 1 else "No heart disease risk predicted"

    return {
        "prediction": prediction,
        "risk_label": label,
        "confidence": round(probability if prediction == 1 else 1 - probability, 4),
        "probability_heart_disease": round(probability, 4),
        "model_name": metadata.get("model_name", "unknown"),
        "model_version": metadata.get("model_version", MODEL_VERSION),
    }
