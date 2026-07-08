import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import build_preprocessor


def sample_training_data():
    rows = [
        [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
        [40, 0, 1, 120, 180, 0, 0, 170, 0, 0.0, 1, 0, 3, 0],
        [65, 1, 0, 150, 260, 1, 1, 120, 1, 2.5, 2, 2, 7, 1],
        [44, 0, 2, 110, 190, 0, 0, 175, 0, 0.2, 1, 0, 3, 0],
        [58, 1, 0, 140, 240, 0, 1, 130, 1, 1.8, 2, 1, 6, 1],
        [49, 0, 1, 118, 210, 0, 0, 165, 0, 0.1, 1, 0, 3, 0],
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN])


def test_preprocessor_output_shape():
    df = sample_training_data()
    x = df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)


def test_training_pipeline_can_fit_and_predict():
    df = sample_training_data()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x, y)
    preds = model.predict(x)
    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1})
