from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UCI_FULL_ARCHIVE_DIR = RAW_DIR / "uci_full_archive"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Official UCI directory for the Heart Disease dataset.
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

# Raw-only policy:
# Download metadata, the unprocessed 76-attribute source files, and the Costs folder.
# The processed.* files are intentionally excluded from the modeling workflow.
UCI_DIRECTORY_FILES = [
    "heart-disease.names",
    "Index",
    "WARNING",
    "ask-detrano",
    "bak",
    "cleveland.data",
    "hungarian.data",
    "switzerland.data",
    "long-beach-va.data",
    "new.data",
    "costs/heart-disease.cost",
    "costs/heart-disease.delay",
    "costs/heart-disease.expense",
    "costs/heart-disease.group",
    "costs/heart-disease.README",
    "costs/Index",
]

# The four raw institutional database files documented by UCI.
# new.data is downloaded for traceability but is not used for training to avoid
# undocumented duplication with the four named institutional sources.
RAW_76_SOURCE_FILES = {
    "cleveland": "cleveland.data",
    "hungarian": "hungarian.data",
    "switzerland": "switzerland.data",
    "long_beach_va": "long-beach-va.data",
}

# Backward-compatible variable name. Not used by the raw-only pipeline.
PROCESSED_SOURCE_FILES: dict[str, str] = {}

# Output files produced by data acquisition.
RAW_DATA_PATH = UCI_FULL_ARCHIVE_DIR / "cleveland.data"
RAW_76_SELECTED_PATH = PROCESSED_DIR / "heart_disease_raw_76_selected14.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "heart_disease_raw_only_all_sites_cleaned.csv"
ALL_PROCESSED_RAW_PATH = PROCESSED_DIR / "not_used_processed_sources_raw_only_policy.csv"
CLEVELAND_ONLY_CLEAN_PATH = PROCESSED_DIR / "heart_disease_raw_only_cleveland_cleaned.csv"
MODEL_PATH = MODEL_DIR / "final_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_SEED = 42
MODEL_VERSION = "v1.2-raw-76-only"
MLFLOW_EXPERIMENT_NAME = "heart-disease-risk-prediction-mlops-raw-76-only"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"
SOURCE_COLUMN = "source_database"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

RAW_COLUMNS = FEATURE_COLUMNS + ["num"]

# 1-based attribute numbers from the UCI documentation for the 14 published ML variables.
# The raw files contain 76 attributes; these 14 are extracted directly from the raw records.
RAW_76_SELECTED_ATTRIBUTE_NUMBERS = [3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51, 58]
RAW_76_RECORD_LENGTH = 76

for path in [RAW_DIR, UCI_FULL_ARCHIVE_DIR, PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, MLRUNS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
