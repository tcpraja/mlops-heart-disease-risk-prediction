from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(CLEAN_DATA_PATH)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    preds = model.predict(x)
    probs = model.predict_proba(x)[:, 1]

    result = {
        "roc_auc_full_cleaned_dataset": float(roc_auc_score(y, probs)),
        "classification_report": classification_report(y, preds, zero_division=0, output_dict=True),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
