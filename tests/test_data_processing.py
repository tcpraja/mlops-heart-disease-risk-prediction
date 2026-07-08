import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_acquisition import clean_heart_data


def test_clean_heart_data_binary_target():
    df = pd.DataFrame(
        [
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 0],
            [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2, 2],
        ],
        columns=FEATURE_COLUMNS + ["num"],
    )
    clean = clean_heart_data(df)
    assert TARGET_COLUMN in clean.columns
    assert list(clean[TARGET_COLUMN]) == [0, 1]
    assert "num" not in clean.columns


def test_clean_heart_data_keeps_feature_columns():
    df = pd.DataFrame(
        [[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1]],
        columns=FEATURE_COLUMNS + ["num"],
    )
    clean = clean_heart_data(df)
    assert list(clean.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
